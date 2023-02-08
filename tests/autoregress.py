import datetime
import time
import traceback
from typing import Union, Dict, Any, Optional, Callable, Tuple
import os 
os.environ["JAX_PLATFORMS"] = ""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import smart_open
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.utils import check_min_version
from flax import jax_utils
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import lax
from jax.experimental.compilation_cache import compilation_cache
from optax import GradientTransformation
from optax._src.numerics import safe_int32_increment
from optax._src.transform import ScaleByAdamState
from transformers import AutoTokenizer, FlaxLongT5Model, T5Tokenizer
import smart_open
from PIL import Image
import matplotlib.pyplot as plt

_RESHAPE = False

def load(path: str, prototype: Dict[str, jax.Array]):
    try:
        with smart_open.open(path + ".np", 'rb') as f:
            params = list(zip(*sorted([(int(i), v) for i, v in np.load(f).items()])))[1]
    except:
        with smart_open.open(path + ".np", 'rb') as f:
            params = \
                list(zip(*sorted([(int(i), v) for i, v in np.load(f, allow_pickle=True)["arr_0"].item().items()])))[1]

    _, tree = jax.tree_util.tree_flatten(prototype)
    return tree.unflatten(params)

def shift(x: jax.Array, amount: int):
        return jnp.concatenate([jnp.zeros_like(x[:amount]),x[amount:]], 0)

@jax.custom_gradient
def communicate(x: jax.Array):
    normalizer = jnp.arange(x.shape[0] * 2).reshape(-1, *(1,) * (x.ndim - 1)) + 1

    def _grad(dy: jax.Array):
        dy0, dy, dy0c, dyc = jnp.split(dy, 4, -1)
        dyc = lax.cumsum(jnp.concatenate([dy0c, dyc], 0) / normalizer, 0, reverse=True)
        dy0c, dyc = jnp.split(dyc, 2, 0)
        dy0 = dy0 + dy0c
        dy0 = lax.select_n(True,dy0, jnp.zeros_like(dy0))
        dy0 = shift(dy0, -1)
        return dy + dy0 + dyc

    x0 = shift(x, 1)
    x0 = lax.select_n(True,x0, jnp.zeros_like(x))
    cat = jnp.concatenate([x0, x], 0)
    cat = jnp.concatenate([cat, lax.cumsum(cat, 0) / normalizer], 0)
    cat = cat.reshape(4, *x.shape).transpose(*range(1, x.ndim), 0, x.ndim).reshape(*x.shape[:-1], -1)
    return cat, _grad

#_original_call = nn.Conv.__call__


def conv_call(self: nn.Conv, inputs: jax.Array) -> jax.Array:
    inputs = jnp.asarray(inputs, self.dtype)
    if _RESHAPE and "quant" not in self.scope.name:
        inputs = communicate(inputs)
    return _original_call(self, inputs)

#nn.Conv.__call__ = conv_call





def patch_weights(weights: Dict[str, Any], do_patch: bool = False):
    new_weights = {}
    for k, v in weights.items():
        if isinstance(v, dict):
            new_weights[k] = patch_weights(v, ("conv" in k and "quant" not in k) or do_patch)
        elif isinstance(v, (list, tuple)):
            new_weights[k] = list(zip(*sorted(patch_weights(dict(enumerate(v)), "conv" in k or do_patch).items())))[1]
        elif isinstance(v, jax.Array) and do_patch and k == "kernel":
            # KernelShape + (in_features,) + (out_features,)
            new_weights[k] = jnp.concatenate([v * 1e-2, v, v * 1e-3, v * 1e-4], -2)
        elif isinstance(v, jax.Array):
            new_weights[k] = v
        else:
            print(f"Unknown type {type(v)}")
            new_weights[k] = v
    return new_weights



base_model = "flax/stable-diffusion-2-1"
base_path = 'Checkpoint/'
t5_tokens: int = 2 ** 13
vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

#vae_params = patch_weights(vae_params)

t5_conv = nn.Sequential([nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                            nn.LayerNorm(epsilon=1e-10),
                            nn.relu,
                            nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                            ])
inp_shape = jax.random.normal(jax.random.PRNGKey(0), (jax.local_device_count(), t5_tokens, 768))
t5_conv_params = t5_conv.init(jax.random.PRNGKey(0), inp_shape)

context = 1

embd = {"embd": np.load(base_path+"embd.np", allow_pickle=True)['0']}


#vae_params = load(base_path + "vae", vae_params)
#unet_params = load(base_path + "unet", unet_params)

t5_conv_params = load(base_path + "conv", t5_conv_params)

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
text_encoder = FlaxLongT5Model.from_pretrained("google/long-t5-tglobal-base", dtype=jnp.float32)


schedule_length = 1024
guidance_scale = 7.5
latent_shape = (1,4,32*context,32)

noise_rng = jax.random.PRNGKey(0)
latents = jax.random.normal(noise_rng, latent_shape)#these can be any shape (64x etc),and it gets 8x'd

noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    num_train_timesteps=1024)
sched_state = noise_scheduler.create_state()
timesteps = 20
sched_state = noise_scheduler.set_timesteps(sched_state,timesteps,latents.shape)#schedule_length=100, if there's still noise increase that 

text = "bla bla bla bla"
t5_tokens = 2 ** 13
text_tokens = tokenizer(text, return_tensors="jax", padding="max_length", max_length=t5_tokens)
encoded = text_encoder.encode(text_tokens['input_ids'], text_tokens['attention_mask'], params=text_encoder.params).last_hidden_state
encoded = t5_conv.apply(t5_conv_params, encoded)[0]
#encoded = np.array(jnp.concatenate([encoded, embd["embd"][:context]], 0))
cond = encoded.reshape(1, *encoded.shape)


text = ""
t5_tokens = 2 ** 13
text_tokens = tokenizer(text, return_tensors="jax", padding="max_length", max_length=t5_tokens)
encoded = text_encoder.encode(text_tokens['input_ids'], text_tokens['attention_mask'], params=text_encoder.params).last_hidden_state
encoded = t5_conv.apply(t5_conv_params, encoded)[0]
#encoded = np.array(jnp.concatenate([encoded, embd["embd"][:context]], 0))
uncond = encoded.reshape(1, *encoded.shape)

encoded =  jnp.concatenate([uncond]+ [cond], 0)

for t in jnp.arange(timesteps)[::1]:

    new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
    noise_pred = unet.apply({"params": unet_params}, new, t, encoded).sample
    uncond, cond = jnp.split(noise_pred, 2, 0)
    noise_pred = uncond + guidance_scale * (cond - uncond)
    latents, sched_state = noise_scheduler.step(sched_state,noise_pred, t, latents).to_tuple()

    print(latents, t)

#let's see what hte fuck we just did
inp = np.array(jnp.transpose(latents, (0, 2, 3, 1))) / 0.18215

_RESHAPE = True
out = jnp.transpose(vae.apply({"params": vae_params}, inp, method=vae.decode).sample,(0, 2, 3, 1))#vae works so problem is with the unet necessairly. specifically, the denoising.
_RESHAPE = False
print(out.shape)
test = np.array(np.split(out[0], context, axis=0))
test.shape
for i in range(context):
    plt.imshow(out[i])
    plt.figure
