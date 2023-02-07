from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import smart_open
import typer
from PIL import Image
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.utils import check_min_version
from flax import linen as nn
from jax import lax
from jax.experimental.compilation_cache import compilation_cache
from transformers import AutoTokenizer, FlaxLongT5Model, T5Tokenizer

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
compilation_cache.initialize_cache("compilation_cache")

_CONTEXT = 0
_RESHAPE = False
_UPLOAD_RETRIES = 8

_original_call = nn.Conv.__call__


def conv_call(self: nn.Conv, x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, self.dtype)
    if _RESHAPE and "quant" not in self.scope.name:
        normalizer = jnp.arange(x.shape[0] * 2).reshape(1, -1, *(1,) * (x.ndim - 1)) + 1
        x = x.reshape(-1, _CONTEXT, x.shape[1:])
        x0 = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], 1)
        cat = jnp.concatenate([x0, x], 1)
        cat = jnp.concatenate([cat, lax.cumsum(cat, 1) / normalizer], 1)
        x = cat.reshape(-1, 4, *x.shape[1:]).transpose(0, *range(2, x.ndim), 1, x.ndim).reshape(*x.shape[:-1], -1)
    return _original_call(self, x)


nn.Conv.__call__ = conv_call


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


def dict_to_array_dispatch(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return dict_to_array_dispatch(v.item())
        if v.dtype == object:
            raise ValueError(str(v))
        return v
    elif isinstance(v, dict):
        return dict_to_array(v)
    elif isinstance(v, (list, tuple)):
        return list(zip(*sorted(dict_to_array(dict(enumerate(v))).items())))[1]
    else:
        return dict_to_array(v)


def dict_to_array(x):
    new_weights = {}
    for k, v in dict(x).items():
        new_weights[k] = dict_to_array_dispatch(v)
    return new_weights


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


@app.command()
def main(context: int = 8, base_model: str = "flax/stable-diffusion-2-1", schedule_length: int = 1024,
         t5_tokens: int = 2 ** 13, base_path: str = "gs://video-us/checkpoint/", iterations: int = 4096,
         devices: int = 128):
    global _CONTEXT, _RESHAPE
    _CONTEXT = context
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

    t5_conv = nn.Sequential([nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                             nn.LayerNorm(epsilon=1e-10),
                             nn.relu,
                             nn.Conv(features=1024, kernel_size=(25,), strides=(8,)),
                             ])

    inp_shape = jax.random.normal(jax.random.PRNGKey(0), (jax.local_device_count(), t5_tokens, 768))
    t5_conv_params = t5_conv.init(jax.random.PRNGKey(0), inp_shape)

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    text_encoder = FlaxLongT5Model.from_pretrained("google/long-t5-tglobal-base", dtype=jnp.float32)
    vae_params = patch_weights(vae_params)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet
    tokenizer: T5Tokenizer = tokenizer
    text_encoder: FlaxLongT5Model = text_encoder

    pos_embd = jax.random.normal(jax.random.PRNGKey(0), (t5_tokens // 64, 1024))
    latent_merge00 = jax.random.normal(jax.random.PRNGKey(0), (1024, 2048))
    latent_merge01 = jax.random.normal(jax.random.PRNGKey(0), (1024, 2048))
    latent_merge01 = latent_merge01 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 1)
    latent_merge1 = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024))
    latent_merge1 = latent_merge1 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 0)
    pos_embd = pos_embd
    external = {"embd": pos_embd, "merge00": latent_merge00, "merge01": latent_merge01, "merge1": latent_merge1}

    vae_params = load(base_path + "vae", vae_params)
    unet_params = load(base_path + "unet", unet_params)
    t5_conv_params = load(base_path + "conv", t5_conv_params)
    external = load(base_path + "embd", external)

    noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    sched_state = noise_scheduler.create_state()

    local_batch = 1
    ctx = context * jax.device_count()

    def get_encoded(input_ids: jax.Array, attention_mask: Optional[jax.Array]):
        input_ids, attention_mask = input_ids.reshape(devices * 8, -1), attention_mask.reshape(devices * 8, -1)
        encoded = text_encoder.encode(input_ids, attention_mask, params=text_encoder.params).last_hidden_state
        encoded = encoded.reshape(devices, -1, 768)  # [batch, t5_tokens, features]
        encoded = t5_conv.apply(t5_conv_params, encoded)
        encoded = encoded.reshape(1, -1, 1024)
        encoded = lax.broadcast_in_dim(encoded, (local_batch, context, *encoded.shape[1:]), (0, 2, 3))
        return encoded.reshape(local_batch * context, encoded.shape[2], -1) + external["embd"].reshape(1, -1, 1024)

    def merge(latent, noise):
        shape = noise.shape
        latent = latent.reshape(latent.shape[0], -1) @ external["merge00"]
        noise = noise.reshape(noise.shape[0], -1) @ external["merge01"]
        if latent.shape[0] != noise.shape[0]:
            latent = lax.broadcast_in_dim(latent, (noise.shape[0] // latent.shape[0], *latent.shape), (1, 2))
            latent = latent.reshape(-1, latent.shape[-1])
        out = jnp.maximum(noise + latent, 0) @ external["merge1"]
        return out.reshape(shape)

    def vae_apply(*args, method=vae.__call__, **kwargs):
        global _RESHAPE
        _RESHAPE = True
        out = vae.apply(*args, method=method, **kwargs)
        _RESHAPE = False
        return out

    def sample(input_ids: jax.Array, attention_mask: jax.Array, guidance: int):
        tokens = input_ids.size
        unc_tok = jnp.concatenate([jnp.ones((1,)), jnp.zeros((tokens - 1,))]).reshape(input_ids.shape)
        no_text = get_encoded(unc_tok, unc_tok)
        text = get_encoded(input_ids, attention_mask)
        encoded = jnp.concatenate([no_text, no_text, no_text, text, no_text, text])

        def _step(original_latents, idx):
            def _inner(state, i):
                latents, state = state
                new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
                inp = original_latents[jnp.maximum(idx - ctx, 0):jnp.maximum(idx, ctx)]
                pred = unet.apply({"params": unet_params}, merge(inp, new), i, encoded).sample
                uncond, cond = jnp.split(pred, 2, 0)
                pred = uncond + guidance * (cond - uncond)
                return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

            state = noise_scheduler.set_timesteps(sched_state, schedule_length, (1, 16, 16, 4))
            latents = jax.random.normal(jax.random.PRNGKey(0), (1, 16, 16, 4), jnp.float32)
            latents = lax.broadcast_in_dim(latents, (3, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
            (out, _), _ = lax.scan(_inner, (latents, state), jnp.arange(schedule_length)[::-1])
            return original_latents.at[idx].set(out[0]), None

        latents = jnp.zeros((iterations, 16, 16, 4))
        (out, _), _ = lax.scan(_step, latents, jnp.arange(schedule_length)[::-1])
        return jnp.transpose(vae_apply({"params": vae_params}, out, method=vae.decode).sample, (0, 2, 3, 1))

    p_sample = jax.jit(sample)

    while True:
        guidance = float(input())
        inp = input()

        toks = tokenizer(inp, return_tensors="np", padding="max_length", truncation=True, max_length=t5_tokens)
        sample_out = p_sample(toks["input_ids"].reshape(devices * 8, -1),
                              toks["attention_mask"].reshape(devices * 8, -1),
                              jnp.full((), guidance, jnp.float32))
        Image.fromarray(sample_out, "RGB").save("out.png")


if __name__ == "__main__":
    app()
