import datetime
import math
import time
import traceback
from typing import Union, Dict, Callable, Optional, List, Tuple, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import smart_open
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.models.attention_flax import FlaxAttentionBlock, FlaxGEGLU
from diffusers.utils import check_min_version
from flax import jax_utils
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import lax
from jax._src.nn import functions as nn_functions
from optax import GradientTransformation
from optax._src.numerics import safe_int32_increment
from optax._src.transform import ScaleByAdamState
from transformers import CLIPTokenizer, FlaxCLIPTextModel

from data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
_UPLOAD_RETRIES = 8
global _SHUFFLE 
_SHUFFLE = False

def attention(query: jax.Array, key: jax.Array, value: jax.Array, scale: float):
    ctx_dims = f'{"b" * (key.ndim > 3)}zhf'

    def _softmax(q: jax.Array, k: jax.Array) -> jax.Array:
        lgt = jnp.einsum(f"bshf,{ctx_dims}->bhsz", q, k) * scale
        lgt = jnp.exp(lgt - lgt.max(-1, keepdims=True))
        return lgt / lgt.sum(-1, keepdims=True)

    @jax.custom_gradient
    def _fn(q: jax.Array, k: jax.Array, v: jax.Array):
        out = jnp.einsum(f"bhsz,{ctx_dims}->bshf", _softmax(q, k), v)

        def _grad(dy: jax.Array):
            dy = dy.reshape(out.shape)
            lgt = _softmax(q, k)
            prod = jnp.einsum(f"bhsz,{ctx_dims},bshf->bhsz", lgt, v, dy) * scale
            d_lgt = prod - prod.sum(-1, keepdims=True) * lgt

            d_v = jnp.einsum(f"bshf,bhsz->{ctx_dims}", dy, lgt)
            d_q = jnp.einsum(f"bhsz,{ctx_dims}->bshf", d_lgt, k)
            d_k = jnp.einsum(f"bhsz,bshf->{ctx_dims}", d_lgt, q)
            return d_q, d_k, d_v

        return out.reshape(*out.shape[:-2], -1), _grad

    return _fn(query, key, value)


def _new_attention(self: FlaxAttentionBlock, hidden_states: jax.Array, context: Optional[jax.Array] = None,
                   deterministic=True):
    context = hidden_states if context is None else context

    query_proj = self.query(hidden_states).reshape(*hidden_states.shape[:-1], self.heads, -1)
    key_proj = self.key(context).reshape(*context.shape[:-1], self.heads, -1)
    value_proj = self.value(context).reshape(*context.shape[:-1], self.heads, -1)
    hidden_states = attention(query_proj, key_proj, value_proj, self.scale)
    return self.proj_attn(hidden_states)


FlaxAttentionBlock.__call__ = _new_attention


_original_call = nn.Conv.__call__

@jax.custom_gradient
def communicate(x: jax.Array):
    def _grad(dy: jax.Array):
        left, mid, right = jnp.split(dy, 3, 1)
        right = lax.ppermute(right, "batch", [(i, (i + 1) % jax.device_count()) for i in range(jax.device_count())])
        left = lax.ppermute(left, "batch", [((i + 1) % jax.device_count(), i) for i in range(jax.device_count())])
        return dy + left + right

    left = lax.ppermute(x, "batch", [(i, (i + 1) % jax.device_count()) for i in range(jax.device_count())])
    right = lax.ppermute(x, "batch", [((i + 1) % jax.device_count(), i) for i in range(jax.device_count())])
    return jnp.concatenate([left, x, right], 1), _grad

def conv_call(self: nn.Conv, inputs: jax.Array) -> jax.Array:
    inputs = jnp.asarray(inputs, self.dtype)
    if _SHUFFLE and "quant" not in self.scope.name:
        inputs = communicate(inputs)
    out = _original_call(self, inputs)
    if _SHUFFLE and "quant" not in self.scope.name:
        _, out, _ = jnp.split(out, 3, 1)
    return out

nn.Conv.__call__ = conv_call


def device_id():
    return lax.axis_index("batch")


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


def _take_0th(x):
    return x[0]


def to_host(k, index_fn: Callable[[jax.Array], jax.Array] = _take_0th):
    return jax.device_get(jax.tree_util.tree_map(index_fn, k))


def promote(inp: jax.Array) -> jax.Array:
    return jnp.asarray(inp, jnp.promote_types(jnp.float64, jnp.result_type(inp)))


def clip_norm(val: jax.Array, min_norm: float) -> jax.Array:
    return jnp.maximum(jnp.sqrt(lax.square(val).sum()), min_norm)


def ema(x, y, beta, step):
    out = (1 - beta) * x + beta * y
    return out / (1 - beta ** step), out



def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule, clip: float = 1e-2) -> GradientTransformation: #adam+lion
    def zero(x):
        return jnp.zeros_like(x, dtype=jnp.bfloat16)
    def init_fn(params):
        return {"momentum": jax.tree_util.tree_map(zero, params), "mu": jax.tree_util.tree_map(zero, params), "nu": jax.tree_util.tree_map(zero, params), "count": jnp.zeros((), dtype=jnp.int64)}

    def update_fn(updates, state, params=None):
        count = state["count"] + 1

        def get_update(grad: jax.Array, param: jax.Array, mom: jax.Array, mu: jax.Array, nu: jax.Array):
            dtype = mom.dtype
            grad, param, mom, nu, mu = jax.tree_map(promote, (grad, param, mom, nu, mu))
            g_norm = clip_norm(grad, 1e-16)
            p_norm = clip_norm(param, 1e-3)
            grad *= lax.min(p_norm / g_norm * clip, 1.)

            nuc, nu = ema(lax.square(grad), nu, b2, count)
            grad /= lax.max(lax.sqrt(nuc), eps)
            muc, mu = ema(grad, mu, b1, count)


            delta = grad - mom
            update = lax.sign(mom + delta * (1 - b1))

            update *= jnp.linalg.norm(muc) / jnp.linalg.norm(update) * -lr(count)
            return update,  (mom + delta * (1 - b2)).astype(dtype), mu.astype(dtype), nu.astype(dtype)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in (params, state["momentum"], state["mu"], state["nu"])]
        updates, mom, mu, nu = [treedef.unflatten(leaf) for leaf in zip(*[get_update(*xs) for xs in zip(*all_leaves)])]
        return updates, {"momentum": mom, "count": count, "mu": mu, "nu": nu}

    return GradientTransformation(init_fn, update_fn)


def deep_replace(d, value):
    if isinstance(d, dict):
        return {k: deep_replace(v, value) for k, v in d.items()}
    return value


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
def main(lr: float = 2e-5, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-16, downloaders: int = 2,
         resolution: int = 256, fps: int = 8, context: int = 8, workers: int = 4, prefetch: int = 1,
         batch_prefetch: int = 4, base_model: str = "flax_base_model", data_path: str = "./urls",
         sample_interval: int = 2048, parallel_videos: int = 60, schedule_length: int = 1024, warmup_steps: int = 1024,
         lr_halving_every_n_steps: int = 2 ** 17, clip_tokens: int = 77, save_interval: int = 2048,
         overwrite: bool = True, base_path: str = "gs://video-us/checkpoint_big_context", local_iterations: int = 4,
         unet_batch: int = 1, device_steps: int = 4):
    global _SHUFFLE
    device_steps = jax.device_count()
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, jax.local_device_count(), prefetch,
                      parallel_videos, tokenizer, clip_tokens, device_steps, batch_prefetch)

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

    text_encoder = FlaxCLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", dtype=jnp.float32)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet

    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")

    if not overwrite:
        unet_params = load(base_path + "unet", unet_params)

    lr_sched = optax.warmup_exponential_decay_schedule(0, lr, warmup_steps, lr_halving_every_n_steps, 0.5)
    optimizer = scale_by_laprop(beta1, beta2, eps, lr_sched)
    
    unet_state = TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    sched_state = noise_scheduler.create_state()
    unconditioned_tokens = tokenizer([""], padding="max_length", max_length=77, return_tensors="np")

    def get_encoded(input_ids: jax.Array, attention_mask: jax.Array):
        return text_encoder(input_ids, attention_mask, params=text_encoder.params)[0]

    def unet_fn(noise, encoded, timesteps, params):
        global _SHUFFLE
        _SHUFFLE = True
        out = unet.apply({"params": params}, lax.stop_gradient(noise), timesteps, lax.stop_gradient(encoded)).sample
        _SHUFFLE = False
        return out

    def vae_apply(*args, method=vae.__call__, **kwargs):
        return vae.apply({"params": vae_params}, *args, method=method, **kwargs)

    def sample_vae(inp: jax.Array):
        return jnp.transpose(vae_apply(inp, method=vae.decode).sample, (0, 2, 3, 1))

    def all_to_all(x, split=1):
        return lax.all_to_all(x.reshape(1, *x.shape), "batch", split, 0, tiled=True)

    def all_to_all_batch(batch: Dict[str, Union[np.ndarray, int]]) -> Dict[str, Union[np.ndarray, int]]:
        return {"pixel_values": all_to_all(batch["pixel_values"], 1),
                "idx": batch["idx"] + jnp.arange(jax.device_count()),
                "input_ids": all_to_all(batch["input_ids"], 1),
                "attention_mask": all_to_all(batch["attention_mask"], 1)}

    def rng(idx: jax.Array):
        return jax.random.PRNGKey(idx * jax.device_count() + device_id())
    
    def rng_syncd(idx: jax.Array):
        return jax.random.PRNGKey(idx)

    def sample(unet_params, batch: Dict[str, Union[np.ndarray, int]]):
        batch = all_to_all_batch(batch)
        batch = jax.tree_map(lambda x: x[0], batch)
        latent_rng, sample_rng, noise_rng = jax.random.split(rng(batch["idx"]), 3)
        step_rng = rng_syncd(batch["idx"])
        inp = jnp.transpose(batch["pixel_values"][0].astype(jnp.float32) / 255, (0, 3, 1, 2))
        posterior = vae_apply(inp, method=vae.encode)

        hidden_mode = posterior.latent_dist.mode()
        latents = jnp.transpose(hidden_mode, (0, 3, 1, 2)) * 0.18215

        encoded = get_encoded(batch["input_ids"], batch["attention_mask"])
        unc = get_encoded(unconditioned_tokens["input_ids"], unconditioned_tokens["attention_mask"])
        encoded = jnp.concatenate([unc] * 4 + [encoded] * 4, 0)

        def _step(state, i):
            latents, state = state
            new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
            unet_pred = unet_fn(new, encoded, i, unet_params)
            u1, u2, u4, u8, c1, c2, c4, c8 = jnp.split(unet_pred, 8, 0)
            pred = jnp.concatenate([c1, u2 + (c2 - u2) * 2, u4 + (c4 - u4) * 4, u8 + (c8 - u8) * 8])
            return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

        lshape = latents.shape
        shape = (1, lshape[1], lshape[0] * lshape[2], lshape[3])
        noise = jax.random.normal(latent_rng, shape, latents.dtype)
        noise = lax.broadcast_in_dim(noise, (4, *noise.shape), (1, 2, 3, 4)).reshape(-1, *noise.shape[1:])
        latents = lax.broadcast_in_dim(latents, (4, *latents.shape), (1, 2, 3, 4)).reshape(*noise.shape)
        state = noise_scheduler.set_timesteps(sched_state, schedule_length, latents.shape)
        start_step = schedule_length
        t0 = jnp.full((), start_step, jnp.int32)
        latents = noise_scheduler.add_noise(sched_state, latents, noise, t0)

        (out, _), _ = lax.scan(_step, (latents, state), jnp.arange(start_step)[::-1])
        out = out.reshape(4, lshape[1], lshape[0], lshape[2], lshape[3])
        out = out.transpose(0, 2, 3, 4, 1) / 0.18215  # NCHW -> NHWC + remove latent folding
        return jnp.concatenate([sample_vae(x) for x in [hidden_mode] + list(out)])

    p_sample = jax.pmap(sample, "batch")

    def distance(x, y):
        dist = x - y
        dist_sq = lax.square(dist).mean()
        dist_abs = lax.abs(dist).mean()
        return dist_sq, dist_abs

    def train_step(unet_state: TrainState, batch: Dict[str, jax.Array]):
        img = batch["pixel_values"].astype(jnp.float32) / 255 
        inp = jnp.transpose(img[0], (0, 3, 1, 2))
        gauss0, drop0 = jax.random.split(rng(batch["idx"] + 1), 2)
        vae_out = vae_apply(inp, rngs={"gaussian": gauss0, "dropout": drop0}, deterministic=False, method=vae.encode)
        encoded = get_encoded(batch["input_ids"], batch["attention_mask"])
        encoded = encoded.reshape(*encoded.shape[1:])  # remove batch dim for einsum

        def compute_loss(unet_params, itr):
            sample_rng, noise_rng = jax.random.split(rng(itr + batch["idx"]), 2)
            step_rng = rng_syncd(itr + batch["idx"])

            latents = jnp.stack([vae_out.latent_dist.sample(r) for r in jax.random.split(sample_rng, unet_batch)])
            latents = latents.reshape(unet_batch, context * latents.shape[2], latents.shape[3], latents.shape[4])
            latents = latents.transpose(0, 3, 1, 2)
            latents = latents * 0.18215

            noise = jax.random.normal(noise_rng, latents.shape)
            t0 = jax.random.randint(step_rng, (unet_batch,), 0, noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(sched_state, latents, noise, t0)

            unet_pred = unet_fn(noisy_latents, encoded, t0, unet_params)

            unet_dist_sq, unet_dist_abs = distance(unet_pred, noise)

            return unet_dist_sq, (unet_dist_sq, unet_dist_abs)

        (loss, scalars), grads = jax.value_and_grad(lambda x: compute_loss(x, 0), has_aux=True)(unet_state.params)

        if local_iterations > 1:
            def _inner(prev, itr):
                grads = jax.value_and_grad(lambda x: compute_loss(x, itr * 257), has_aux=True)(unet_state.params)
                return jax.tree_util.tree_map(lambda x, y: x / local_iterations + y, grads, prev), None

            prev = jax.tree_util.tree_map(lambda x: x / local_iterations, ((loss, scalars), grads))
            ((loss, scalars), grads), _ = lax.scan(_inner, prev, jnp.arange(1, local_iterations))

        scalars, grads = lax.pmean((scalars, grads), "batch")
        new_unet_state = unet_state.apply_gradients(grads=grads)
        return new_unet_state, scalars

    def train_loop(states, batch: Dict[str, Union[np.ndarray, int]]):
        return lax.scan(train_step, states, all_to_all_batch(batch))

    p_train_step = jax.pmap(train_loop, "batch", donate_argnums=(0, 1))

    unet_state = jax_utils.replicate(unet_state)

    start_time = time.time()

    def to_img(x: jax.Array) -> wandb.Image:
        return wandb.Image(x.reshape(-1, resolution, 3))

    global_step = 0
    for epoch in range(10 ** 9):
        for i, (vid, ids, msk) in tqdm.tqdm(enumerate(data, 1)):

            global_step += 1
            if global_step <= 2:
                print(f"Step {global_step}", datetime.datetime.now())
            i *= device_steps
            batch = {
                "pixel_values": vid,#device steps, batch (ordered), context, resolution, resolution, 3
                "input_ids": ids,#device_steps, batch (ordered), input_ids
                "attention_mask": msk,
                "idx": jnp.full((jax.local_device_count(),jax.device_count()),i)}#device_steps, batch (ordered), input_ids

            extra = {}
            pid = f'{jax.process_index() * context * jax.local_device_count()}-{(jax.process_index() + 1) * context * jax.local_device_count() - 1}'
            if i % sample_interval == 0:
                sample_out = p_sample(unet_state.params, batch)
                s_mode, g1, g2, g4, g8 = np.split(to_host(sample_out, lambda x: x), 5, 1)
                print(f'sample shape {g1.shape}')    
                extra[f"Samples/Reconstruction (Mode) {pid}"] = to_img(s_mode)
                extra[f"Samples/Reconstruction (U-Net, Guidance 1) {pid}"] = to_img(g1)
                extra[f"Samples/Reconstruction (U-Net, Guidance 2) {pid}"] = to_img(g2)
                extra[f"Samples/Reconstruction (U-Net, Guidance 4) {pid}"] = to_img(g4)
                extra[f"Samples/Reconstruction (U-Net, Guidance 8) {pid}"] = to_img(g8)

            print("Before train step", datetime.datetime.now())
            unet_state, scalars = p_train_step(unet_state, batch)
            print("After train step", datetime.datetime.now())

            timediff = time.time() - start_time
            sclr = to_host(scalars)
            print("To host", datetime.datetime.now())

            for offset, (unet_sq, unet_abs) in enumerate(zip(*sclr)):
                print("loop step", datetime.datetime.now())
                vid_per_day = i / timediff * 24 * 3600 * jax.device_count()
                log = {"U-Net MSE/Total": float(unet_sq), "U-Net MAE/Total": float(unet_abs),
                       "Step": i + offset - device_steps, "Epoch": epoch}
                if offset == device_steps - 1:
                    log.update(extra)
                    log.update({"Runtime": timediff, "Speed/Videos per Day": vid_per_day,
                                "Speed/Frames per Day": vid_per_day * context})
                run.log(log, step=(global_step - 1) * device_steps + offset)
            if i % save_interval == 0 and jax.process_index() == 0:
                states = ("unet", unet_state),
                for n, s in list(states):
                    p = to_host(s.params)
                    flattened, jax_structure = jax.tree_util.tree_flatten(p)
                    for _ in range(_UPLOAD_RETRIES):
                        try:
                            with smart_open.open(base_path + n + ".np", "wb") as f:
                                np.savez(f, **{str(i): v for i, v in enumerate(flattened)})
                            break
                        except:
                            print("failed to write", n, "checkpoint")
                            traceback.print_exc()

if __name__ == "__main__":
    app()
