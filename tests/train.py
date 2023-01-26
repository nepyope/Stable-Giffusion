import os
import time
from typing import Union, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxDDPMScheduler
from diffusers.utils import check_min_version
from flax import jax_utils
from flax import linen as nn
from flax.training import train_state
from jax import lax
from jax.experimental.compilation_cache import compilation_cache
from optax import GradientTransformation
from optax._src.numerics import safe_int32_increment
from optax._src.transform import ScaleByAdamState
from transformers import CLIPTokenizer, FlaxCLIPTextModel

from data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
compilation_cache.initialize_cache("compilation_cache")

_CONTEXT = 0
_KERNEL = 3
_RESHAPE = False


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""


_original_call = nn.Conv.__call__


def conv_call(self, inputs: jax.Array) -> jax.Array:
    shape = inputs.shape

    inputs = jnp.asarray(inputs, self.dtype)
    kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
        if x is None:
            x = 1
        if isinstance(x, int):
            return (x,) * len(kernel_size)
        return x

    if inputs.ndim == len(kernel_size) + 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    strides = maybe_broadcast(self.strides)  # self.strides or (1,) * (inputs.ndim - 2)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    quant_reshape = _RESHAPE and "quant" not in self.scope.name

    if quant_reshape:
        inputs = inputs.reshape(-1, _CONTEXT, *shape[1:])

    padding = self.padding
    if self.padding == 'CIRCULAR':
        kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
        pads = [(0, 0)] + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
        inputs = jnp.pad(inputs, pads, mode='wrap')
        padding = 'VALID'

    if isinstance(self.padding, str):
        pad_shape = inputs.shape[int(quant_reshape):]
        ndim = len(pad_shape)
        lhs_perm = (0, ndim - 1) + tuple(range(1, ndim - 1))
        rhs_perm = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
        rhs_shape = np.take(pad_shape, rhs_perm)[2:]
        effective_rhs_shape = [(k - 1) * r + 1 for k, r in zip(rhs_shape, kernel_dilation)]
        padding = lax.padtype_to_pads(np.take(pad_shape, lhs_perm)[2:], effective_rhs_shape, strides, padding)

    if quant_reshape and "values_added" not in self.__dict__:
        # We need to change self.strides while keeping the rest of self intact, so that _original_call can use
        # both the new values (such as the patched strides) and the old values (scope).
        # Unfortunately, Flax overwrites __setattr_, so we can't set self.strides manually. instead, we have to
        # circumvent their __setattr__ assertions (which are sensible in almost all cases!) by manually updating the
        # object's __dict__. After the update, self.strides will have the new value, so that _original_call can use it.
        self.__dict__.update({"values_added": True, "padding": ((_KERNEL - 1, 0),) + tuple(padding),
                              "strides": (1,) + tuple(strides), "input_dilation": (1,) + tuple(input_dilation),
                              "kernel_size": (3,) + tuple(kernel_size)})

    y = _original_call(self, inputs)

    if quant_reshape:
        return y.reshape(shape[0], *y.shape[2:])
    return y


nn.Conv.__call__ = conv_call


def patch_weights(weights: Dict[str, Any], do_patch: bool = False):
    new_weights = {}
    scale = jnp.where(jnp.arange(_KERNEL) == (_KERNEL - 1), 1, 1e-3)
    for k, v in weights.items():
        if isinstance(v, dict):
            new_weights[k] = patch_weights(v, ("conv" in k and "quant" not in k) or do_patch)
        elif isinstance(v, (list, tuple)):
            new_weights[k] = list(zip(*sorted(patch_weights(dict(enumerate(v)), "conv" in k or do_patch).items()))[1])
        elif isinstance(v, jax.Array) and do_patch and k == "kernel":
            new_weights[k] = jnp.stack([v] * _KERNEL, 0) * scale.reshape(-1, *(1,) * v.ndim)
        elif isinstance(v, jax.Array):
            new_weights[k] = v
        else:
            print(f"Unknown type {type(v)}")
            new_weights[k] = v
    return new_weights


def to_host(k):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], k))


def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_map(lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def scale_by_laprop(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> GradientTransformation:
    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = update_moment(updates, state.nu, b2, 2)
        count_inc = safe_int32_increment(state.count)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = jax.tree_map(lambda m, v: m / lax.max(jnp.sqrt(v), eps), updates, nu_hat)
        mu = update_moment(updates, state.mu, b1, 1)
        mu_hat = bias_correction(mu, b1, count_inc)
        return mu_hat, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


@app.command()
def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-12,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1",
         kernel: int = 3, data_path: str = "./urls", batch_size: int = jax.local_device_count(),
         sample_interval: int = 64, parallel_videos: int = 128,
         tracing_start_step: int = 128, tracing_stop_step: int = 196,
         schedule_length: int = 1024):
    global _KERNEL, _CONTEXT, _RESHAPE
    _CONTEXT, _KERNEL = context, kernel
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = FlaxCLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", dtype=jnp.float32)
    vae_params = patch_weights(vae_params)

    vae: FlaxAutoencoderKL = vae
    unet: FlaxUNet2DConditionModel = unet
    tokenizer: CLIPTokenizer = tokenizer
    text_encoder: FlaxCLIPTextModel = text_encoder

    run = wandb.init(entity="homebrewnlp", project="stable-giffusion")

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                            scale_by_laprop(beta1, beta2, eps),
                            # optax.transform.add_decayed_weights(weight_decay, mask),  # TODO: mask normalization
                            optax.sgd(lr),
                            )

    vae_state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)
    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=schedule_length)
    noise_scheduler.create_state()
    local_batch = batch_size // jax.local_device_count()
    mask = jnp.arange(context).reshape(1, -1, 1, 1, 1, 1) > jnp.arange(context).reshape(1, 1, -1, 1, 1, 1)

    def get_encoded(latents: jax.Array, batch: Dict[str, jax.Array]):
        encoded = text_encoder(batch["input_ids"], batch["attention_mask"], params=text_encoder.params)[0]
        encoded = encoded.reshape(local_batch, 1, *encoded.shape[1:])
        encoded = lax.broadcast_in_dim(encoded, (local_batch, context, *encoded.shape[2:]), (0, 1, 2, 3))
        encoded = encoded.reshape(local_batch * context, encoded.shape[2], -1)
        latents = latents.reshape(local_batch, context, 1, *latents.shape[1:])
        latents = lax.broadcast_in_dim(latents, (local_batch, context, context, *latents.shape[3:]),
                                       (0, 1, 2, 3, 4, 5))
        latents = latents * mask
        latents = latents.reshape(local_batch * context, -1, encoded.shape[2])
        return jnp.concatenate([encoded, latents], 1)

    def vae_apply(*args, **kwargs):
        global _RESHAPE
        _RESHAPE = True
        out = vae.apply(*args, **kwargs)
        _RESHAPE = False
        return out

    def sample_vae(params: Any, inp: jax.Array):
        return jnp.transpose(vae_apply({"params": params}, inp, method=vae.decode).sample, (0, 2, 3, 1))

    def sample(unet_params, vae_params, batch: Dict[str, Union[np.ndarray, int]]):
        latent_rng, sample_rng, noise_rng, step_rng = jax.random.split(jax.random.PRNGKey(batch["idx"]), 4)

        inp = jnp.transpose(batch["pixel_values"].astype(jnp.float32) / 255, (0, 3, 1, 2))
        posterior = vae_apply({"params": vae_params}, inp, method=vae.encode)

        hidden_states_rng = posterior.latent_dist.sample(sample_rng)
        hidden_states_mode = posterior.latent_dist.mode()

        latents = jnp.transpose(hidden_states_rng, (0, 3, 1, 2)) * 0.18215
        encoded = get_encoded(latents, batch)

        def _step(state, i):
            unet_pred = unet.apply({"params": unet_params}, state, i, encoded).sample
            return noise_scheduler.add_noise(state, -unet_pred, i), None

        out, _ = lax.scan(_step, jax.random.normal(latent_rng, latents.shape, latents.dtype),
                          jnp.arange(schedule_length))
        out = jnp.transpose(out, (0, 2, 3, 1)) / 0.18215

        sample_rng = sample_vae(vae_params, hidden_states_rng)
        sample_mode = sample_vae(vae_params, hidden_states_mode)
        sample_out = sample_vae(vae_params, out)

        return sample_rng, sample_mode, sample_out

    p_sample = jax.pmap(sample, "batch")

    def distance(x, y):
        dist = (x - y).reshape(local_batch, context, -1)
        dist_sq = lax.pmean(lax.square(dist).mean((0, 2)), "batch")
        dist_abs = lax.pmean(lax.abs(dist).mean((0, 2)), "batch")
        return dist_sq, dist_abs

    def train_step(unet_state: train_state.TrainState, vae_state: train_state.TrainState,
                   batch: Dict[str, Union[np.ndarray, int]]):
        def compute_loss(params):
            unet_params, vae_params = params
            gaussian, dropout, sample_rng, noise_rng, step_rng = jax.random.split(jax.random.PRNGKey(batch["idx"]), 5)

            img = batch["pixel_values"].astype(jnp.float32) / 255
            inp = jnp.transpose(img, (0, 3, 1, 2))
            vae_outputs = vae_apply({"params": vae_params}, inp, deterministic=True, method=vae.encode)
            latents = vae_outputs.latent_dist.sample(sample_rng)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = lax.stop_gradient(latents * 0.18215)

            noise = jax.random.normal(noise_rng, latents.shape)
            timesteps = jax.random.randint(step_rng, (latents.shape[0],), 0, noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoded = get_encoded(latents, batch)
            unet_pred = unet.apply({"params": unet_params}, noisy_latents, timesteps, encoded).sample

            vae_pred = vae_apply({"params": vae_params}, inp, rngs={"gaussian": gaussian, "dropout": dropout},
                                 sample_posterior=True, deterministic=False).sample
            vae_pred = jnp.transpose(vae_pred, (0, 2, 3, 1))

            # TODO: use perceptual loss
            unet_dist_sq, unet_dist_abs = distance(unet_pred, noise)
            vae_dist_sq, vae_dist_abs = distance(vae_pred, img)
            return unet_dist_sq.mean() + vae_dist_sq.mean(), (unet_dist_sq, unet_dist_abs, vae_dist_sq, vae_dist_abs)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, scalars), (unet_grad, vae_grad) = grad_fn((unet_state.params, vae_state.params))
        unet_grad = lax.pmean(unet_grad, "batch")
        vae_grad = lax.pmean(vae_grad, "batch")
        new_unet_state = unet_state.apply_gradients(grads=unet_grad)
        new_vae_state = vae_state.apply_gradients(grads=vae_grad)
        return new_unet_state, new_vae_state, scalars

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))

    vae_state = jax_utils.replicate(vae_state)
    unet_state = jax_utils.replicate(unet_state)

    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, batch_size, prefetch, parallel_videos,
                      tokenizer)
    start_time = time.time()
    for epoch in range(100):
        for i, (video, input_ids, attention_mask) in tqdm.tqdm(enumerate(data, 1)):
            batch = {"pixel_values": video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
                     "idx": jnp.full((jax.local_device_count(),), i, jnp.int32),
                     "input_ids": input_ids.reshape(jax.local_device_count(), -1, *input_ids.shape[1:]),
                     "attention_mask": attention_mask.reshape(jax.local_device_count(), -1, *attention_mask.shape[1:])}
            extra = {}
            if i % sample_interval == 0:
                s_rng, s_mode, s_unet = to_host(p_sample(unet_state.params, vae_state.params, batch))
                extra["Samples/Reconstruction (RNG)"] = wandb.Image(s_rng.reshape(-1, resolution, 3))
                extra["Samples/Reconstruction (Mode)"] = wandb.Image(s_mode.reshape(-1, resolution, 3))
                extra["Samples/Reconstruction (U-Net)"] = wandb.Image(s_unet.reshape(-1, resolution, 3))
                extra["Samples/Ground Truth"] = wandb.Image(batch["pixel_values"][0].reshape(-1, resolution, 3) / 255)

            unet_state, vae_state, scalars = p_train_step(unet_state, vae_state, batch)
            unet_dist_sq, unet_dist_abs, vae_dist_sq, vae_dist_abs = to_host(scalars)
            timediff = time.time() - start_time
            run.log({"U-Net MSE/Total": float(np.mean(unet_dist_sq)), "U-Net MAE/Total": float(np.mean(unet_dist_abs)),
                     "VAE MSE/Total": float(np.mean(vae_dist_sq)), "VAE MAE/Total": float(np.mean(vae_dist_abs)),
                     **{f"U-Net MSE/Frame {k}": float(loss) for k, loss in enumerate(unet_dist_sq)},
                     **{f"U-Net MAE/Frame {k}": float(loss) for k, loss in enumerate(unet_dist_abs)},
                     **{f"VAE MSE/Frame {k}": float(loss) for k, loss in enumerate(vae_dist_sq)},
                     **{f"VAE MAE/Frame {k}": float(loss) for k, loss in enumerate(vae_dist_abs)},
                     **extra,
                     "Step": i, "Runtime": timediff,
                     "Speed/Videos per Day": i * batch_size / timediff * 24 * 3600,
                     "Speed/Frames per Day": i * batch_size * context / timediff * 24 * 3600})

            if i == tracing_start_step:
                jax.profiler.start_trace("trace")
            if i == tracing_stop_step:
                jax.profiler.stop_trace()
        with open("out.np", "wb") as f:
            np.savez(f, **to_host(vae_state.params))


if __name__ == "__main__":
    app()
