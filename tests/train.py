import datetime
import time
import traceback
from typing import Union, Dict, Any, Callable, Tuple

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

from data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
compilation_cache.initialize_cache("compilation_cache")

_CONTEXT = 0
_RESHAPE = False
_UPLOAD_RETRIES = 8

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


def shift(x: jax.Array, amount: int):
    return lax.ppermute(x, "batch", [((i + amount) % jax.device_count(), i) for i in range(jax.device_count())])


@jax.custom_gradient
def communicate(x: jax.Array):
    normalizer = jnp.arange(x.shape[0] * 2).reshape(-1, *(1,) * (x.ndim - 1)) + 1

    def _grad(dy: jax.Array):
        dy0, dy, dy0c, dyc = jnp.split(dy, 4, -1)
        dyc = lax.cumsum(jnp.concatenate([dy0c, dyc], 0) / normalizer, 0, reverse=True)
        dy0c, dyc = jnp.split(dyc, 2, 0)
        dy0 = dy0 + dy0c
        dy0 = lax.select_n(device_id() == 0, dy0, jnp.zeros_like(dy0))
        dy0 = shift(dy0, -1)
        return dy + dy0 + dyc

    x0 = shift(x, 1)
    x0 = lax.select_n(device_id() == 0, x0, jnp.zeros_like(x))
    cat = jnp.concatenate([x0, x], 0)
    cat = jnp.concatenate([cat, lax.cumsum(cat, 0) / normalizer], 0)
    cat = cat.reshape(4, *x.shape).transpose(*range(1, x.ndim), 0, x.ndim).reshape(*x.shape[:-1], -1)
    return cat, _grad


def conv_call(self: nn.Conv, inputs: jax.Array) -> jax.Array:
    inputs = jnp.asarray(inputs, self.dtype)
    if _RESHAPE and "quant" not in self.scope.name:
        inputs = communicate(inputs)
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


def clip_norm(val: jax.Array, min_norm: float) -> jax.Array:
    return jnp.maximum(jnp.sqrt(lax.square(val).sum()), min_norm)


def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule, clip: float = 1e-3) -> GradientTransformation:
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        updates = jax.tree_map(promote, updates)
        params = jax.tree_map(promote, params)
        nu = jax.tree_map(promote, state.nu)
        mu = jax.tree_map(promote, state.mu)

        g_norm = jax.tree_util.tree_map(lambda x: clip_norm(x, 1e-16), updates)
        p_norm = jax.tree_util.tree_map(lambda x: clip_norm(x, 1e-3), params)
        updates = jax.tree_util.tree_map(lambda x, pn, gn: x * lax.min(pn / gn * clip, 1.), updates, p_norm, g_norm)

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
def main(lr: float = 1e-5, beta1: float = 0.95, beta2: float = 0.95, eps: float = 1e-16, downloaders: int = 2,
         resolution: int = 128, fps: int = 4, context: int = 8, workers: int = 16, prefetch: int = 6,
         base_model: str = "flax/stable-diffusion-2-1", data_path: str = "./urls", sample_interval: int = 1024,
         parallel_videos: int = 128, tracing_start_step: int = 3, tracing_stop_step: int = 5,
         schedule_length: int = 1024, guidance: float = 7.5, warmup_steps: int = 16384,
         lr_halving_every_n_steps: int = 2 ** 17, t5_tokens: int = 2 ** 13, pos_embd_scale: float = 1e-3,
         save_interval: int = 2048, overwrite: bool = True, unet_mode: bool = False,
         base_path: str = "gs://video-us/checkpoint/", unet_init_steps: int = 1024, conv_init_steps: int = 0,
         local_iterations: int = 16):
    global _CONTEXT, _RESHAPE
    unet_init_steps -= conv_init_steps * unet_mode
    conv_init_steps *= 1 - unet_mode
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)

    vae_params = patch_weights(vae_params)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet

    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")

    pos_embd = jax.random.normal(jax.random.PRNGKey(0), (context * jax.device_count(), 1024))
    latent_merge0 = jax.random.normal(jax.random.PRNGKey(0), (1024, 2048)) * pos_embd_scale
    latent_merge0 = latent_merge0 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 1)
    latent_merge1 = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024)) * pos_embd_scale
    latent_merge1 = latent_merge1 + jnp.concatenate([jnp.eye(1024), -jnp.eye(1024)], 0)
    pos_embd = pos_embd * pos_embd_scale
    external = {"embd": pos_embd, "merge0": latent_merge0, "merge1": latent_merge1}

    if unet_mode:
        vae_params = load(base_path + "vae", vae_params)

    if not overwrite:
        vae_params = load(base_path + "vae", vae_params)
        unet_params = load(base_path + "unet", unet_params)
        external = load(base_path + "embd", external)

    lr_sched = optax.warmup_exponential_decay_schedule(0, lr, warmup_steps, lr_halving_every_n_steps, 0.5)
    optimizer = scale_by_laprop(beta1, beta2, eps, lr_sched)
    vae_state = TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)
    external_state = TrainState.create(apply_fn=lambda: None, params=external, tx=optimizer)
    unet_state = TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    sched_state = noise_scheduler.create_state()

    local_batch = 1
    full_context = context * jax.device_count()

    def encode(latent, params):
        latent = latent.reshape(context, 1024) @ params["merge0"]
        out = jnp.maximum(latent, 0) @ params["merge1"]
        ctx = device_id() * context
        out += lax.dynamic_slice_in_dim(params["embd"], ctx, context)
        out = lax.all_gather(out, "batch", tiled=True)
        out = lax.broadcast_in_dim(out, (context, full_context, 1024), (1, 2))
        out *= (jnp.arange(context) + ctx).reshape(-1, 1, 1) > jnp.arange(full_context).reshape(1, -1, 1)
        return out

    def unet_fn(noise, encoded, timesteps, unet_params):
        return unet.apply({"params": unet_params}, noise, timesteps, encoded).sample

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
        return {"input_ids": all_to_all(batch["input_ids"]), "attention_mask": all_to_all(batch["attention_mask"]),
                "pixel_values": all_to_all(batch["pixel_values"], 1),
                "idx": batch["idx"] + jnp.arange(jax.device_count())}

    def rng(idx: jax.Array):
        return jax.random.PRNGKey(idx * jax.device_count() + device_id())

    def sample(params, batch: Dict[str, Union[np.ndarray, int]]):
        if unet_mode:
            vae_params = vae_state.params
            unet_params, external_params = params
        else:
            unet_params, vae_params, external_params = params

        batch = all_to_all_batch(batch)
        batch = jax.tree_map(lambda x: x[0], batch)
        latent_rng, sample_rng, noise_rng, step_rng = jax.random.split(rng(batch["idx"]), 4)

        inp = jnp.transpose(batch["pixel_values"].astype(jnp.float32) / 255, (0, 3, 1, 2))
        posterior = vae_apply({"params": vae_params}, inp, method=vae.encode)

        hidden_mode = posterior.latent_dist.mode()
        latents = jnp.transpose(hidden_mode, (0, 3, 1, 2)) * 0.18215

        encoded = encode(latents, external_params)

        def _step(state, i):
            latents, state = state
            new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
            unet_pred = unet_fn(new, encoded, i, unet_params)
            uncond, cond = jnp.split(unet_pred, 2, 0)
            pred = uncond + guidance * (cond - uncond)
            return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

        latents = jax.random.normal(latent_rng, latents.shape, latents.dtype)
        latents = lax.broadcast_in_dim(latents, (3, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
        state = noise_scheduler.set_timesteps(sched_state, schedule_length, latents.shape)
        (out, _), _ = lax.scan(_step, (latents, state), jnp.arange(schedule_length)[::-1])
        out = jnp.transpose(out, (0, 2, 3, 1)) / 0.18215
        nvt, vnt, vt = out.split(3, 0)
        return jnp.concatenate([sample_vae(vae_params, x) for x in (hidden_mode, nvt, vnt, vt)])

    p_sample = jax.pmap(sample, "batch")

    def distance(x, y):
        dist = (x - y).reshape(local_batch, context, -1)
        dist_sq = lax.square(dist).mean()
        dist_abs = lax.abs(dist).mean()
        return dist_sq, dist_abs

    def train_step(all_states: Union[
        Tuple[TrainState, TrainState, TrainState], Tuple[TrainState, TrainState, TrainState, TrainState]],
                   batch: Dict[str, jax.Array]):
        if unet_mode:
            unet_state, external_state = all_states
            v_state = vae_state
        else:
            unet_state, v_state, external_state = all_states

        if unet_mode:
            img = batch["pixel_values"].astype(jnp.float32) / 255
            inp = jnp.transpose(img, (0, 3, 1, 2))
            vae_out = vae_apply({"params": vae_params}, inp, deterministic=True, method=vae.encode)

        def compute_loss(params, itr):
            if unet_mode:
                vae_params = vae_state.params
                unet_params, external_params = params
            else:
                unet_params, vae_params, external_params = params
            itr = (itr + rng(batch["idx"])).astype(jnp.uint32)
            gauss0, gauss1, drop0, drop1, sample_rng, noise_rng, step_rng = jax.random.split(itr, 7)

            if unet_mode:
                vae_outputs = vae_out
            else:
                vae_outputs = vae_apply({"params": vae_params}, inp, rngs={"gaussian": gauss0, "dropout": drop0},
                                        deterministic=False, method=vae.encode)

            vae_outputs = vae_outputs.latent_dist.sample(sample_rng)
            latents = jnp.transpose(vae_outputs, (0, 3, 1, 2))
            latents = lax.stop_gradient(latents * 0.18215)

            noise = jax.random.normal(noise_rng, latents.shape)
            timesteps = jax.random.randint(step_rng, (latents.shape[0],), 0, noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(sched_state, latents, noise, timesteps)

            encoded = encode(latents, external_params)
            unet_pred = unet_fn(noisy_latents, encoded, timesteps, unet_params)

            # TODO: use perceptual loss
            unet_dist_sq, unet_dist_abs = distance(unet_pred, noise)

            if unet_mode:
                vae_dist_sq = vae_dist_abs = jnp.zeros(())
            else:
                vae_pred = vae_apply({"params": vae_params}, vae_outputs[:context],
                                     rngs={"gaussian": gauss1, "dropout": drop1}, deterministic=False,
                                     method=vae.decode).sample
                vae_pred = jnp.transpose(vae_pred, (0, 2, 3, 1))
                vae_dist_sq, vae_dist_abs = distance(vae_pred, img)

            return unet_dist_sq.mean() + vae_dist_sq.mean(), (unet_dist_sq, unet_dist_abs, vae_dist_sq, vae_dist_abs)

        if unet_mode:
            inp = (unet_state.params, external_state.params)
        else:
            inp = (unet_state.params, v_state.params, external_state.params)

        (loss, scalars), grads = jax.value_and_grad(lambda x: compute_loss(x, 0), has_aux=True)(inp)
        if local_iterations > 1:
            def _inner(state, itr):
                prev = state
                grad_fn = jax.value_and_grad(lambda x: compute_loss(x, itr * 2 ** 20), has_aux=True)
                return jax.tree_util.tree_map(lambda x, y: x / local_iterations + y, grad_fn(inp), prev), None

            ((loss, scalars), grads), _ = lax.scan(_inner, ((loss, scalars), grads), jnp.arange(1, local_iterations))

        scalars, grads = lax.pmean((scalars, grads), "batch")
        new_unet_state = lax.switch((batch["idx"] > unet_init_steps).astype(jnp.int32),
                                    [lambda: unet_state, lambda: unet_state.apply_gradients(grads=grads[0])])
        grads[-1]["embd"] = grads[-1]["embd"] * 0.1  # embedding gradient shrink from GLM
        new_external_state = lax.switch((batch["idx"] > conv_init_steps).astype(jnp.int32), [lambda: external_state,
                                                                                             lambda: external_state.apply_gradients(
                                                                                                 grads=grads[-1])])
        if unet_mode:
            return (new_unet_state, new_external_state), scalars

        new_vae_state = v_state.apply_gradients(grads=grads[1])
        return (new_unet_state, new_vae_state, new_external_state), scalars

    def train_loop(states, batch: Dict[str, Union[np.ndarray, int]]):
        return lax.scan(train_step, states, all_to_all_batch(batch))

    p_train_step = jax.pmap(train_loop, "batch", donate_argnums=(0, 1))

    if not unet_mode:
        vae_state = jax_utils.replicate(vae_state)
    unet_state = jax_utils.replicate(unet_state)
    external_state = jax_utils.replicate(external_state)

    data = DataLoader(workers, data_path, downloaders, resolution, fps, context * jax.device_count(),
                      jax.local_device_count(), prefetch, parallel_videos)
    start_time = time.time()

    def to_img(x: jax.Array) -> wandb.Image:
        return wandb.Image(x.reshape(-1, resolution, 3))

    global_step = 0
    for epoch in range(10 ** 9):
        for i, video in tqdm.tqdm(enumerate(data, 1)):
            global_step += 1
            if global_step <= 2:
                print(f"Step {global_step}", datetime.datetime.now())
            i *= jax.device_count()
            batch = {"pixel_values": video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
                     "idx": jnp.full((jax.local_device_count(),), i, jnp.int32)}
            extra = {}
            pid = f'{jax.process_index() * context * jax.local_device_count()}-{(jax.process_index() + 1) * context * jax.local_device_count() - 1}'
            if i % sample_interval == 0:
                if unet_mode:
                    params = unet_state.params, external_state.params
                else:
                    params = unet_state.params, vae_state.params, external_state.params
                sample_out = p_sample(params, batch)
                s_mode, s_nvt, s_vnt, s_vt = np.split(to_host(sample_out, lambda x: x), 4, 1)
                extra[f"Samples/Reconstruction (Mode) {pid}"] = to_img(s_mode)
                extra[f"Samples/Reconstruction (U-Net, Text Guidance) {pid}"] = to_img(s_nvt)
                extra[f"Samples/Reconstruction (U-Net, Video Guidance) {pid}"] = to_img(s_vnt)
                extra[f"Samples/Reconstruction (U-Net, Full Guidance) {pid}"] = to_img(s_vt)
                extra[f"Samples/Ground Truth {pid}"] = to_img(batch["pixel_values"].astype(jnp.float32) / 255)

            if unet_mode:
                (unet_state, external_state), scalars = p_train_step((unet_state, external_state), batch)
            else:
                (unet_state, vae_state, external_state), scalars = p_train_step((unet_state, vae_state, external_state),
                                                                                batch)

            timediff = time.time() - start_time
            for offset, (unet_sq, unet_abs, vae_sq, vae_abs) in enumerate(zip(*to_host(scalars))):
                vid_per_day = i / timediff * 24 * 3600
                log = {"U-Net MSE/Total": float(unet_sq), "U-Net MAE/Total": float(unet_abs),
                       "VAE MSE/Total": float(vae_sq), "VAE MAE/Total": float(vae_abs),
                       "Step": i + offset - jax.device_count(), "Epoch": epoch}
                if offset == jax.device_count() - 1:
                    log.update(extra)
                    log.update({"Runtime": timediff, "Speed/Videos per Day": vid_per_day,
                                "Speed/Frames per Day": vid_per_day * context * jax.device_count()})
                run.log(log, step=(global_step - 1) * jax.device_count() + offset)
            if i == tracing_start_step * jax.device_count():
                jax.profiler.start_trace("trace")
            if i == tracing_stop_step * jax.device_count():
                jax.profiler.stop_trace()
            if i % save_interval == 0 and jax.process_index() == 0:
                states = ("unet", unet_state), ("embd", external_state)
                for n, s in [("vae", vae_state)] * (not unet_mode) + list(states):
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
