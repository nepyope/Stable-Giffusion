import datetime
import operator
import os
from typing import Union, Dict, Any

import jax
import numpy as np
import optax
import typer
from diffusers import FlaxAutoencoderKL
from diffusers.utils import check_min_version
from flax import jax_utils, linen as nn
from flax.training import train_state
from flax.training.common_utils import shard
from jax import lax, numpy as jnp

from data import DataLoader

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
_CONTEXT = 0
_KERNEL = 3
_RESHAPE = False


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Conv3d(nn.Conv):
    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        shape = inputs.shape
        if _RESHAPE:
            inputs = inputs.reshape(-1, _CONTEXT, *shape[1:])

        inputs = jnp.asarray(inputs, self.dtype)

        if isinstance(self.kernel_size, int):
            raise TypeError('The kernel size must be specified as a'
                            ' tuple/list of integers (eg.: [3, 3]).')
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        strides = maybe_broadcast(self.strides)  # self.strides or (1,) * (inputs.ndim - 2)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        in_features = inputs.shape[-1]
        assert in_features % self.feature_group_count == 0
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count, self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)

        padding = self.padding
        if self.padding == 'CIRCULAR':
            kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
            pads = [(0, 0)] + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
            inputs = jnp.pad(inputs, pads, mode='wrap')
            padding = 'VALID'

        if isinstance(self.padding, str):
            lhs_perm, rhs_perm, _ = _conv_dimension_numbers(inputs.shape[1:])
            rhs_shape = np.take(inputs.shape[1:], rhs_perm)[2:]
            effective_rhs_shape = [(k - 1) * r + 1 for k, r in zip(rhs_shape, kernel_dilation)]
            padding = lax.padtype_to_pads(np.take(inputs.shape[1:], lhs_perm)[2:], effective_rhs_shape, strides,
                                          padding)
        else:
            padding = tuple((operator.index(lo), operator.index(hi)) for lo, hi in padding)

        padding = ((_KERNEL - 1, 0),) + padding

        y = lax.conv_general_dilated(inputs, kernel, strides, padding,
                                     lhs_dilation=input_dilation, rhs_dilation=kernel_dilation,
                                     dimension_numbers=_conv_dimension_numbers(inputs.shape),
                                     feature_group_count=self.feature_group_count,
                                     precision=self.precision)
        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        if _RESHAPE:
            return y.reshape(shape[0], *y.shape[2:])
        return y


nn.Conv = Conv3d


def patch_weights(weights: Dict[str, Any], do_patch: bool = False):
    new_weights = {}
    scale = jnp.where(jnp.arange(_KERNEL) == (_KERNEL - 1), 1, 1e-3)
    for k, v in weights.items():
        if isinstance(v, dict):
            new_weights[k] = patch_weights(v, "conv" in k or do_patch)
        elif isinstance(v, (list, tuple)):
            new_weights[k] = list(zip(*sorted(patch_weights(dict(enumerate(v)), "conv" in k or do_patch).items()))[1])
        elif isinstance(v, jax.Array) and do_patch and k == "kernel":
            new_weights[k] = jnp.stack([v] * _KERNEL, 0) * scale.reshape(-1, *(1,) * (v.ndim))
        elif isinstance(v, jax.Array):
            new_weights[k] = v
        else:
            print(f"Unknown type {type(v)}")
            new_weights[k] = v
    return new_weights


def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-12,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1",
         kernel: int = 3, data_path: str = "./urls"):
    global _KERNEL, _CONTEXT, _RESHAPE
    _CONTEXT, _KERNEL = context, kernel
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32,
                                                        use_auth_token=True)
    vae_params = patch_weights(vae_params)
    vae: FlaxAutoencoderKL = vae

    adamw = optax.adamw(learning_rate=optax.constant_schedule(lr), b1=beta1, b2=beta2, eps=eps,
                        weight_decay=weight_decay)

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)

    state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)

    _RESHAPE = True

    def train_step(state: train_state.TrainState, batch: Dict[str, Union[np.ndarray, int]]):
        def compute_loss(params):
            vae_outputs = vae.apply({"params": params}, batch["pixel_values"], method=vae.encode)
            # Later on simply add previous latent and embed(timestamp) as "text embeddings" to the unet
            latents = vae_outputs.latent_dist.sample(jax.random.PRNGKey(batch["idx"]))
            out = vae.apply({"params": params}, latents, method=vae.decode)
            loss = lax.square(out - batch["pixel_values"]).mean()  # TODO: Use perceptual loss
            return lax.pmean(loss, "batch")

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grad)
        return new_state, loss

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))
    state = jax_utils.replicate(state)
    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, jax.local_device_count(), prefetch)
    for epoch in range(100):
        for i, video in enumerate(data):
            batch = shard({"pixel_values": video, "idx": i})
            state, loss = p_train_step(state, batch)
            print(datetime.datetime.now(), epoch, i, loss)
        with open("out.np", "wb") as f:
            np.savez(f, **jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params)))


if __name__ == "__main__":
    typer.run(main)
