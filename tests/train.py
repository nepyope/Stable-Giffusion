import datetime
import json
import os
import time
from typing import Union, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.utils import check_min_version
from flax import jax_utils
from flax import linen as nn
from flax.training import train_state
from jax import lax
from jax.experimental.compilation_cache import compilation_cache
from optax import GradientTransformation
from optax._src.numerics import safe_int32_increment
from optax._src.transform import ScaleByAdamState
from transformers import AutoTokenizer, T5Tokenizer

from data import DataLoader

jax.distributed.initialize()
app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
compilation_cache.initialize_cache("compilation_cache")

_CONTEXT = 0
_RESHAPE = False

_original_call = nn.Conv.__call__


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def device_id():
    return lax.axis_index("batch")


def conv_call(self: nn.Conv, inputs: jax.Array) -> jax.Array:
    inputs = jnp.asarray(inputs, self.dtype)
    #if _RESHAPE and "quant" not in self.scope.name:
    #    i0 = lax.ppermute(inputs, "batch", [(i, (i + 1) % jax.device_count()) for i in range(jax.device_count())])
    #    arange = jnp.arange(inputs.shape[0]).reshape(-1, *(1,) * (inputs.ndim - 1))
    #    inputs = jnp.concatenate([inputs, i0, lax.cumsum(inputs, 0) / arange], -1)
    return _original_call(self, inputs)


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
            new_weights[k] = jnp.concatenate([v, v * 1e-2, v * 1e-3], -2)
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


def _take_0th(x):
    return x[0]


def to_host(k, index_fn: Callable[[jax.Array], jax.Array] = _take_0th):
    return jax.device_get(jax.tree_util.tree_map(index_fn, k))


def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_map(lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def promote(inp: jax.Array) -> jax.Array:
    return jnp.asarray(inp, jnp.promote_types(jnp.float64, jnp.result_type(inp)))


def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule) -> GradientTransformation:
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_map(promote, updates)
        nu = jax.tree_map(promote, state.nu)
        mu = jax.tree_map(promote, state.mu)
        nu = update_moment(updates, nu, b2, 2)
        count_inc = safe_int32_increment(state.count)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = jax.tree_map(lambda m, v: m / lax.max(lax.sqrt(v), eps), updates, nu_hat)
        mu = update_moment(updates, mu, b1, 1)
        scale = -lr(count_inc)
        mu_hat = bias_correction(mu, b1, count_inc)
        mu_hat = jax.tree_map(lambda x: x * scale, mu_hat)
        mu = jax.tree_map(lambda x, o: x.astype(o.dtype), mu, state.mu)
        nu = jax.tree_map(lambda x, o: x.astype(o.dtype), nu, state.nu)
        return mu_hat, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


def deep_replace(d, value):
    if isinstance(d, dict):
        return {k: deep_replace(v, value) for k, v in d.items()}
    return value


@app.command()
def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-16,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1",
         data_path: str = "./urls", sample_interval: int = 64, parallel_videos: int = 128,
         tracing_start_step: int = 128, tracing_stop_step: int = 196,
         schedule_length: int = 1024,
         guidance: float = 7.5,
         unet_batch_factor: int = 16,
         warmup_steps: int = 2048,
         lr_halving_every_n_steps: int = 8192,
         t5_tokens: int = 2 ** 13,
         save_interval: int = 1024,
         overwrite: bool = False):
    global _CONTEXT, _RESHAPE
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    # vae_params = patch_weights(vae_params)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet
    tokenizer: T5Tokenizer = tokenizer

    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")

    lr_sched = optax.warmup_exponential_decay_schedule(0, lr, warmup_steps, lr_halving_every_n_steps, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                            scale_by_laprop(beta1, beta2, eps, lr_sched),
                            # optax.transform.add_decayed_weights(weight_decay, mask),  # TODO: mask normalization
                            )

    if not overwrite and os.path.isfile("vae.np"):
        vae_params = list(zip(*sorted(np.load("vae.np").items())))[1]
        unet_params = list(zip(*sorted(np.load("unet.np").items())))[1]
        with open("vae.json", 'r') as f:
            _, structure = jax.tree_util.tree_flatten(deep_replace(json.load(f), jnp.zeros((1,))))
        vae_params = structure.unflatten(vae_params)
        with open("unet.json", 'r') as f:
            _, structure = jax.tree_util.tree_flatten(deep_replace(json.load(f), jnp.zeros((1,))))
        unet_params = structure.unflatten(unet_params)
        with open("conv.json", 'r') as f:
            _, structure = jax.tree_util.tree_flatten(deep_replace(json.load(f), jnp.zeros((1,))))

    vae_state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)
    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    sched_state = noise_scheduler.create_state()

    local_batch = 1

    def get_encoded(latents: jax.Array):
        latents = latents.reshape(-1, context, *latents.shape[1:])
        latents = lax.all_gather(latents, "batch", axis=1, tiled=True)

        latents = lax.broadcast_in_dim(latents, (latents.shape[0], context, *latents.shape[1:]),
                                       (0, 2, 3, 4, 5))
        mask = jnp.arange(context * jax.device_count()).reshape(1, 1, -1, 1, 1, 1)
        mask = (jnp.arange(context).reshape(1, -1, 1, 1, 1, 1) + device_id() * context) > mask
        latents = latents * mask
        return latents.reshape(latents.shape[0] * context, -1, 1024)

    def vae_apply(*args, method=vae.__call__, **kwargs):
        global _RESHAPE
        _RESHAPE = True
        out = vae.apply(*args, method=method, **kwargs)
        _RESHAPE = False
        return out

    def sample_vae(params: Any, inp: jax.Array):
        return jnp.transpose(vae_apply({"params": params}, inp, method=vae.decode).sample, (0, 2, 3, 1))

    def all_to_all(x, split=2):
        return lax.all_to_all(x.reshape(1, *x.shape), "batch", split, 0, tiled=True)

    def all_to_all_batch(batch: Dict[str, Union[np.ndarray, int]]) -> Dict[str, Union[np.ndarray, int]]:
        return {"pixel_values": all_to_all(batch["pixel_values"], 1),
                "idx": batch["idx"] + jnp.arange(jax.device_count())}

    def rng(idx: jax.Array):
        return jax.random.PRNGKey(idx * jax.device_count() + device_id())

    def sample(unet_params, vae_params, batch: Dict[str, Union[np.ndarray, int]]):
        batch = all_to_all_batch(batch)
        batch = jax.tree_map(lambda x: x[0], batch)
        latent_rng, sample_rng, noise_rng, step_rng = jax.random.split(rng(batch["idx"]), 4)

        inp = jnp.transpose(batch["pixel_values"].astype(jnp.float32) / 255, (0, 3, 1, 2))
        posterior = vae_apply({"params": vae_params}, inp, method=vae.encode)

        hidden_mode = posterior.latent_dist.mode()
        latents = jnp.transpose(hidden_mode, (0, 3, 1, 2)) * 0.18215
        vid_text = get_encoded(latents)
        no_vid_text = get_encoded(jnp.zeros_like(latents))
        encoded = jnp.concatenate([no_vid_text, vid_text])

        def _step(state, i):
            latents, state = state
            new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])

            unet_pred = unet.apply({"params": unet_params}, new, i, encoded).sample
            uncond, cond = jnp.split(unet_pred, 2, 0)
            pred = uncond + guidance * (cond - uncond)
            return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

        latents = jax.random.normal(latent_rng, latents.shape, latents.dtype)
        state = noise_scheduler.set_timesteps(sched_state, schedule_length, latents.shape)
        (out, _), _ = lax.scan(_step, (latents, state), jnp.arange(schedule_length)[::-1])
        out = jnp.transpose(out, (0, 2, 3, 1)) / 0.18215
        return jnp.concatenate([sample_vae(vae_params, x) for x in (hidden_mode, out)])

    p_sample = jax.pmap(sample, "batch")

    def distance(x, y):
        dist = (x - y).reshape(local_batch, context, -1)
        dist_sq = lax.square(dist).mean()
        dist_abs = lax.abs(dist).mean()
        return dist_sq, dist_abs

    def train_step(all_states, batch: Dict[str, Union[np.ndarray, int]]):
        unet_state, vae_state = all_states

        def compute_loss(params):
            unet_params, vae_params = params
            gaussian, dropout, sample_rng, noise_rng, step_rng = jax.random.split(rng(batch["idx"]), 5)

            img = batch["pixel_values"].astype(jnp.float32) / 255
            inp = jnp.transpose(img, (0, 3, 1, 2))
            vae_outputs = vae_apply({"params": vae_params}, inp, deterministic=True, method=vae.encode)
            latents = jnp.concatenate([vae_outputs.latent_dist.sample(r)
                                       for r in jax.random.split(sample_rng, unet_batch_factor)])
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = lax.stop_gradient(latents * 0.18215)

            noise = jax.random.normal(noise_rng, latents.shape)
            timesteps = jax.random.randint(step_rng, (latents.shape[0],), 0, noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(sched_state, latents, noise, timesteps)

            encoded = get_encoded(latents)
            unet_pred = unet.apply({"params": unet_params}, noisy_latents, timesteps, encoded).sample

            vae_pred = vae_apply({"params": vae_params}, inp, rngs={"gaussian": gaussian, "dropout": dropout},
                                 sample_posterior=True, deterministic=False).sample
            vae_pred = jnp.transpose(vae_pred, (0, 2, 3, 1))

            # TODO: use perceptual loss
            unet_dist_sq, unet_dist_abs = distance(unet_pred, noise)
            vae_dist_sq, vae_dist_abs = distance(vae_pred, img)
            return unet_dist_sq.mean() + vae_dist_sq.mean(), (unet_dist_sq, unet_dist_abs, vae_dist_sq, vae_dist_abs)

        compute_loss = jax.remat(compute_loss, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, scalars), (unet_grad, vae_grad) = grad_fn((unet_state.params, vae_state.params))
        unet_grad = lax.pmean(unet_grad, "batch")
        vae_grad = lax.pmean(vae_grad, "batch")
        new_unet_state = unet_state.apply_gradients(grads=unet_grad)
        new_vae_state = vae_state.apply_gradients(grads=vae_grad)
        return (new_unet_state, new_vae_state), lax.pmean(scalars, "batch")

    def train_loop(unet_state: train_state.TrainState, vae_state: train_state.TrainState,
                   batch: Dict[str, Union[np.ndarray, int]]):
        return lax.scan(train_step, (unet_state, vae_state), all_to_all_batch(batch))

    p_train_step = jax.pmap(train_loop, "batch", donate_argnums=(0, 1))

    vae_state = jax_utils.replicate(vae_state)
    unet_state = jax_utils.replicate(unet_state)

    data = range(100)
    start_time = time.time()

    def to_img(x: jax.Array) -> wandb.Image:
        return wandb.Image(x.reshape(-1, resolution, 3))

    first_step = True
    for epoch in range(10 ** 9):
        for i, video in tqdm.tqdm(enumerate(data, 1)):
            if first_step:
                print("got first data sample", datetime.datetime.now())
            video = np.ones((jax.local_device_count() * context * jax.device_count(), resolution, resolution, 3))
            i *= jax.process_count()
            batch = {"pixel_values": video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
                     "idx": jnp.full((jax.local_device_count(),), i, jnp.int32)}
            extra = {}
            pid = f'{jax.process_index() * context * jax.local_device_count()}-{(jax.process_index() + 1) * context * jax.local_device_count() - 1}'
            if i % sample_interval == 0:
                sample_out = p_sample(unet_state.params, vae_state.params, batch)
                s_mode, s_vnt = np.split(to_host(sample_out, lambda x: x), 2, 1)
                extra[f"Samples/Reconstruction (Mode) {pid}"] = to_img(s_mode)
                extra[f"Samples/Reconstruction (U-Net, Video Guided) {pid}"] = to_img(s_vnt)
                extra[f"Samples/Ground Truth {pid}"] = to_img(batch["pixel_values"].astype(jnp.float32) / 255)

            (unet_state, vae_state), scalars = p_train_step(unet_state, vae_state, batch)
            timediff = time.time() - start_time
            for offset, (unet_sq, unet_abs, vae_sq, vae_abs) in enumerate(zip(*to_host(scalars))):
                vid_per_day = (i + jax.process_count()) / timediff * 24 * 3600
                log = {"U-Net MSE/Total": float(unet_sq), "U-Net MAE/Total": float(unet_abs),
                       "VAE MSE/Total": float(vae_sq), "VAE MAE/Total": float(vae_abs),
                       "Step": i + offset, "Epoch": epoch}
                if offset == jax.device_count() - 1:
                    log.update(extra)
                    log.update({"Runtime": timediff, "Speed/Videos per Day": vid_per_day,
                                "Speed/Frames per Day": vid_per_day * context * jax.process_count()})
                run.log(log, step=i + offset)
            if i == tracing_start_step:
                jax.profiler.start_trace("trace")
            if i == tracing_stop_step:
                jax.profiler.stop_trace()
            if i % save_interval == 0:
                for n, s in (("vae", vae_state), ("unet", unet_state), ("conv", t5_conv_state)):
                    p = to_host(s.params)
                    flattened, jax_structure = jax.tree_util.tree_flatten(p)
                    structure = str(jax_structure)  # like "PyTreeDef({'2': {'a': *}})"
                    structure = structure.replace('PyTreeDef', '')[1:-1]  # clean up "types"
                    structure = structure.replace(': *', ': null').replace("{'", '{"').replace("':", '":')
                    structure = structure.replace("', ", '", ').replace(", '", ', "')  # to valid JSON
                    with open(n + '.json', 'w') as f:
                        f.write(structure)
                    with open(n + ".np", "wb") as f:
                        np.savez(f, **dict(enumerate(flattened)))


if __name__ == "__main__":
    app()
