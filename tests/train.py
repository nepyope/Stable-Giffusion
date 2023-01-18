import os
import time
from typing import Union, Dict, Any

import jax
import numpy as np
import optax
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL
from diffusers.utils import check_min_version
from flax import jax_utils, linen as nn
from flax.training import train_state
from jax import lax, numpy as jnp

from data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
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


@app.command()
def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-12,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1",
         kernel: int = 3, data_path: str = "./urls", batch_size: int = jax.local_device_count()):
    global _KERNEL, _CONTEXT, _RESHAPE
    _CONTEXT, _KERNEL = context, kernel
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32,
                                                        use_auth_token=True)
    vae_params = patch_weights(vae_params)
    vae: FlaxAutoencoderKL = vae
    run = wandb.init(project="stable-giffusion")

    adamw = optax.adamw(learning_rate=optax.constant_schedule(lr), b1=beta1, b2=beta2, eps=eps,
                        weight_decay=weight_decay)

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)

    state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)

    _RESHAPE = True

    def train_step(state: train_state.TrainState, batch: Dict[str, Union[np.ndarray, int]]):
        def compute_loss(params):
            inp = jnp.transpose(batch["pixel_values"], (0, 3, 1, 2))
            out = vae.apply({"params": params}, inp).sample
            out = jnp.transpose(out, (0, 2, 3, 1))

            # TODO: use perceptual loss
            dist = out - batch["pixel_values"]
            dist = dist.reshape(-1, context, *dist.shape[1:])
            dist_sq = lax.pmean(lax.square(dist).mean((0, 2, 3, 4)), "batch")
            dist_abs = lax.pmean(dist.abs().mean((0, 2, 3, 4)), "batch")
            return dist_sq.mean(), (dist_sq, dist_abs)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (dist_sq, dist_abs)), grad = grad_fn(state.params)
        grad = lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        return new_state, (dist_sq, dist_sq.mean(), dist_abs, dist_abs.mean())

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))
    state = jax_utils.replicate(state)
    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, batch_size, prefetch)
    start_time = time.time()
    for epoch in range(100):
        for i, video in tqdm.tqdm(enumerate(data, 1)):
            batch = {"pixel_values": video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
                     "idx": jnp.full((jax.local_device_count(),), i, jnp.int32)}
            state, scalars = p_train_step(state, batch)
            (sq, sq_m, ab, ab_m) = to_host(scalars)
            timediff = time.time() - start_time
            run.log({"MSE/Total": float(sq_m[0]), "MAE/Total": float(ab_m),
                     **{f"MSE/Frame {k}": float(loss) for k, loss in enumerate(sq)},
                     **{f"MAE/Frame {k}": float(loss) for k, loss in enumerate(ab)},
                     "Step": i, "Runtime": timediff,
                     "Speed/Videos per Day": i * batch_size / timediff,
                     "Speed/Frames per Day": i * batch_size * context / timediff})
        with open("out.np", "wb") as f:
            np.savez(f, **to_host(state.params))


if __name__ == "__main__":
    app()
