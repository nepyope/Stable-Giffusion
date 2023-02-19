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

from old_data import DataLoader

app = typer.Typer(pretty_exceptions_enable=False)
check_min_version("0.10.0.dev0")
_UPLOAD_RETRIES = 8


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


def _canonicalize_axes(rank: int, axes: List[int]) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


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


def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule, clip: float = 1e-2) -> GradientTransformation:
    def init_fn(params):
        return ScaleByAdamState(mu=jax.tree_map(jnp.zeros_like, params),  # First Moment
                                nu=jax.tree_map(jnp.zeros_like, params),  # Second Moment
                                count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        count = safe_int32_increment(state.count)

        def get_update(grad: jax.Array, param: jax.Array, nu: jax.Array, mu: jax.Array):
            dtype = nu.dtype
            grad, param, nu, mu = jax.tree_map(promote, (grad, param, nu, mu))
            g_norm = clip_norm(grad, 1e-16)
            p_norm = clip_norm(param, 1e-3)
            grad *= lax.min(p_norm / g_norm * clip, 1.)

            nuc, nu = ema(lax.square(grad), nu, b2, count)
            grad /= lax.max(lax.sqrt(nuc), eps)
            muc, mu = ema(grad, mu, b1, count)
            return muc * -lr(count), nu.astype(dtype), mu.astype(dtype)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in (params, state.nu, state.mu)]
        updates, nu, mu = [treedef.unflatten(leaf) for leaf in zip(*[get_update(*xs) for xs in zip(*all_leaves)])]
        return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule, clip: float = 1e-2) -> GradientTransformation:
    def init_fn(params):
        return {"momentum": jax.tree_util.tree_map(jnp.zeros_like, params), "count": jnp.zeros((), dtype=jnp.int64)}

    def update_fn(updates, state, params=None):
        count = state["count"] + 1

        def get_update(grad: jax.Array, param: jax.Array, mom: jax.Array):
            dtype = mom.dtype
            grad, param, mom = jax.tree_map(promote, (grad, param, mom))
            g_norm = clip_norm(grad, 1e-16)
            p_norm = clip_norm(param, 1e-3)
            grad *= lax.min(p_norm / g_norm * clip, 1.)

            delta = grad - mom
            update = mom + delta * (1 - b1)
            return update * -lr(count), (mom + delta * (1 - b2)).astype(dtype)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in (params, state.nu, state.mu)]
        updates, mom = [treedef.unflatten(leaf) for leaf in zip(*[get_update(*xs) for xs in zip(*all_leaves)])]
        return jnp.sign(updates), {"momentum": mom, "count": count}

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
         resolution: int = 128, fps: int = 1, context: int = 8, workers: int = 8, prefetch: int = 8,
         batch_prefetch: int = 2, base_model: str = "flax_base_model", data_path: str = "./urls",
         sample_interval: int = 2048, parallel_videos: int = 128, schedule_length: int = 1024, warmup_steps: int = 1024,
         lr_halving_every_n_steps: int = 2 ** 17, clip_tokens: int = 77, save_interval: int = 2048,
         overwrite: bool = True, base_path: str = "gs://video-us/checkpoint/", local_iterations: int = 4,
         unet_batch: int = 1, device_steps: int = 4):
    
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, jax.local_device_count(), prefetch,
                      parallel_videos, tokenizer, clip_tokens, device_steps, batch_prefetch)

    global_step = 0
    for epoch in range(10 ** 9):
        for i, (vid, ids, msk) in tqdm.tqdm(enumerate(data, 1)):
            print(f'{vid.shape} {ids.shape} {msk.shape}')


if __name__ == "__main__":
    app()
