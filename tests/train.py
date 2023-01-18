import datetime
import os
from typing import Union, Dict

import jax
import numpy as np
import optax
import typer
from diffusers import FlaxAutoencoderKL
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from jax import lax, numpy as jnp

from data import DataLoader

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def main(lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99, weight_decay: float = 0.001, eps: float = 1e-12,
         max_grad_norm: float = 1, downloaders: int = 4, resolution: int = 384, fps: int = 4, context: int = 16,
         workers: int = os.cpu_count() // 2, prefetch: int = 2, base_model: str = "flax/stable-diffusion-2-1"):
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32,
                                                        use_auth_token=True)
    vae: FlaxAutoencoderKL = vae

    adamw = optax.adamw(learning_rate=optax.constant_schedule(lr),
                        b1=beta1,
                        b2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        )

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)

    state = train_state.TrainState.create(apply_fn=vae.__call__, params=vae_params, tx=optimizer)

    def train_step(state: train_state.TrainState, batch: Dict[str, Union[np.ndarray, int]]):
        def compute_loss(params):
            vae_outputs = vae.apply({"params": params}, batch["pixel_values"], method=vae.encode)
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
    data = DataLoader(workers, "./data", downloaders, resolution, fps, context, jax.local_device_count(), prefetch)
    for epoch in range(100):
        for i, video in enumerate(data):
            batch = shard({"pixel_values": video, "idx": i})
            state, loss = p_train_step(state, batch)
            print(datetime.datetime.now(), epoch, i, loss)
        with open("out.np", "wb") as f:
            np.savez(f, **jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params)))


if __name__ == "__main__":
    typer.run(main)
