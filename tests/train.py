import math
import hashlib
import copy
import datetime
import operator
import time
import traceback
from typing import Union, Dict, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import smart_open
import tqdm
import typer
import wandb
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxPNDMScheduler
from diffusers.models.attention_flax import FlaxAttentionBlock
from flax import jax_utils
from flax.training.train_state import TrainState
from jax import lax
from optax import GradientTransformation
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers import CLIPTokenizer, FlaxCLIPTextModel

from data import DataLoader

cc.initialize_cache("/home/ubuntu/cache")
app = typer.Typer(pretty_exceptions_enable=False)
_UPLOAD_RETRIES = 8
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


def rotate(left: jax.Array, right: jax.Array):
    return (lax.ppermute(left, "batch", [(i, (i + 1) % jax.device_count()) for i in range(jax.device_count())]),
            lax.ppermute(right, "batch", [((i + 1) % jax.device_count(), i) for i in range(jax.device_count())]))

@jax.custom_gradient
def communicate(x: jax.Array):
    if not _SHUFFLE:
        return x, lambda y: y

    def _grad(dy: jax.Array):
        mid, left, right = jnp.split(dy, 3, -1)
        right, left = rotate(right, left)
        return mid + left + right

    left, right = rotate(x, x)
    return jnp.concatenate([x, left, right], -1), _grad


def _new_attention(self: FlaxAttentionBlock, hidden_states: jax.Array, context: Optional[jax.Array] = None,
                   deterministic=True):
    hidden_states = communicate(hidden_states)
    context = hidden_states if context is None else context  # context is pre-communicated

    query_proj = self.query(hidden_states).reshape(*hidden_states.shape[:-1], self.heads, -1)
    key_proj = self.key(context).reshape(*context.shape[:-1], self.heads, -1)
    value_proj = self.value(context).reshape(*context.shape[:-1], self.heads, -1)
    hidden_states = attention(query_proj, key_proj, value_proj, self.scale)
    return self.proj_attn(communicate(hidden_states))


FlaxAttentionBlock.__call__ = _new_attention


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


def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule,
                    clip: float = 1e-2) -> GradientTransformation:  # adam+lion
    def zero(x):
        return jnp.zeros_like(x, dtype=jnp.bfloat16)

    def init_fn(params):
        return {"mu": jax.tree_util.tree_map(zero, params),
                "nu": jax.tree_util.tree_map(zero, params), "count": jnp.zeros((), dtype=jnp.int64)}

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

            update = lax.sign(muc)

            update *= jnp.linalg.norm(muc) / jnp.linalg.norm(update) * -lr(count)
            return update, mu.astype(dtype), nu.astype(dtype)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in (params, state["mom"], state["mu"], state["nu"])]
        updates, mu, nu = [treedef.unflatten(leaf) for leaf in zip(*[get_update(*xs) for xs in zip(*all_leaves)])]
        return updates, {"count": count, "mu": mu, "nu": nu}

    return GradientTransformation(init_fn, update_fn)


def load(path: str, prototype: Dict[str, jax.Array]):
    try:
        with smart_open.open(path + ".np", 'rb') as f:
            params = list(zip(*sorted([(int(i), v) for i, v in np.load(f).items()])))[1]
    except:
        with smart_open.open(path + ".np", 'rb') as f:
            params = list(zip(*sorted([(int(i), v)
                                       for i, v in np.load(f, allow_pickle=True)["arr_0"].item().items()])))[1]

    _, tree = jax.tree_util.tree_flatten(prototype)
    return tree.unflatten(params)


def log(*args, **kwargs):
    print(f'{datetime.datetime.now()} | ', *args, **kwargs)


def compile_fn(fn, name: str):
    log(f"Starting to compile: {name}")
    out = fn()
    log(f"Finished compilation of: {name}")
    return out


@app.command()
def main(lr: float = 1e-6, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-16, downloaders: int = 2,
         resolution: int = 256, fps: int = 8, context: int = 8, workers: int = 4, prefetch: int = 1,
         batch_prefetch: int = 4, base_model: str = "flax_base_model", data_path: str = "./urls",
         sample_interval: int = 2048, parallel_videos: int = 60, schedule_length: int = 1024, warmup_steps: int = 1024,
         lr_halving_every_n_steps: int = 2 ** 17, clip_tokens: int = 77, save_interval: int = 2048,
         overwrite: bool = True, base_path: str = "gs://video-us/checkpoint_2", local_iterations: int = 6,
         unet_batch: int = 1, video_group: int = 8, subsample: int = 32):
    lr *= jax.device_count() ** 0.5
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    data = DataLoader(workers, data_path, downloaders, resolution, fps, context, jax.local_device_count() * video_group, prefetch,
                      parallel_videos, tokenizer, clip_tokens, jax.device_count(), batch_prefetch)

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(base_model, subfolder="vae", dtype=jnp.float32)

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(base_model, subfolder="unet", dtype=jnp.float32)
    for _, v0 in unet_params.items():
        for k1, v1 in v0.items():
            if not k1.startswith("attentions_"):
                continue
            for k2, v2 in v1.items():
                if not k2.startswith("transformer_blocks_"):
                    continue
                for k3, v3 in v2.items():
                    if not k3.startswith("attn"):
                        continue
                    for k4, v4 in v3.items():
                        v4["kernel"] = jnp.concatenate([v4["kernel"]] + [v4["kernel"] * 0.01] * 2, 0)

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
        global _SHUFFLE
        out = text_encoder(input_ids, attention_mask, params=text_encoder.params)[0]
        _SHUFFLE = True
        out = communicate(out)
        _SHUFFLE = False
        return out

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
        out = lax.all_to_all(x.reshape(1, *x.shape), "batch", split, 0, tiled=True)
        return out.reshape(jax.device_count() * video_group, -1, *out.shape[2:])

    def all_to_all_batch(batch: Dict[str, Union[np.ndarray, int]]) -> Dict[str, Union[np.ndarray, int]]:
        return {"pixel_values": all_to_all(batch["pixel_values"], 1),
                "idx": batch["idx"] + jnp.arange(jax.device_count() * video_group),
                "input_ids": all_to_all(batch["input_ids"], 1),
                "attention_mask": all_to_all(batch["attention_mask"], 1)}

    def rng(idx: jax.Array):
        return jax.random.PRNGKey(idx * jax.device_count() + device_id())

    def rng_synced(idx: jax.Array):
        return jax.random.PRNGKey(idx)

    lshape = []
    hidden_dtype = None

    def encode_for_sampling(batch: Dict[str, Union[np.ndarray, int]]):
        nonlocal lshape
        nonlocal hidden_dtype

        batch = all_to_all_batch(batch)

        batch = jax.tree_map(lambda x: x[0], batch)
        inp = jnp.transpose(batch["pixel_values"][0].astype(jnp.float32) / 255, (0, 3, 1, 2))
        posterior = vae_apply(inp, method=vae.encode)

        hidden_mode = posterior.latent_dist.mode()

        encoded = get_encoded(batch["input_ids"], batch["attention_mask"])
        unc = get_encoded(unconditioned_tokens["input_ids"], unconditioned_tokens["attention_mask"])
        encoded = jnp.concatenate([unc, encoded], 0)

        lshape = hidden_mode.shape[0], hidden_mode.shape[3], hidden_mode.shape[1], hidden_mode.shape[2]
        hidden_dtype = hidden_mode.dtype

        return sample_vae(hidden_mode), encoded

    def sample(unet_params, encoded: jax.Array):
        encoded = lax.broadcast_in_dim(encoded, (encoded.shape[0], 4, *encoded.shape[1:]),
                                       (0, *range(2, encoded.ndim + 1)))
        encoded = jnp.reshape(encoded, (-1, *encoded.shape[2:]))

        def _step(state, i):
            latents, state = state
            new = lax.broadcast_in_dim(latents, (2, *latents.shape), (1, 2, 3, 4)).reshape(-1, *latents.shape[1:])
            unet_pred = unet_fn(new, encoded, i, unet_params)
            u, c = jnp.split(unet_pred, 2, 0)
            pred = u + (c - u) * 2 ** jnp.arange(1, 5).reshape(-1, 1, 1, 1)
            return noise_scheduler.step(state, pred, i, latents).to_tuple(), None

        shape = (lshape[1], lshape[0] * lshape[2], lshape[3])
        noise = jax.random.normal(rng(0), shape, hidden_dtype)
        noise = lax.broadcast_in_dim(noise, (4, *shape), (1, 2, 3))
        state = noise_scheduler.set_timesteps(sched_state, schedule_length, noise.shape)

        (out, _), _ = lax.scan(_step, (noise, state), jnp.arange(schedule_length)[::-1])
        out = out.reshape(4, lshape[1], lshape[0], lshape[2], lshape[3])
        out = out.transpose(0, 2, 3, 4, 1) / 0.18215  # NCHW -> NHWC + remove latent folding
        out = out.reshape(-1, *out.shape[2:])
        return sample_vae(out)

    def distance(x, y):
        dist = x - y
        dist_sq = lax.square(dist).mean()
        dist_abs = lax.abs(dist).mean()
        return dist_sq / jax.device_count() ** 2, dist_abs / jax.device_count() ** 2

    def train_step(outer_state: TrainState, batch: Dict[str, jax.Array]):
        batch = all_to_all_batch(batch)

        def _vae_apply(_, b):
            img = b["pixel_values"].astype(jnp.float32) / 255
            inp = jnp.transpose(img[0], (0, 3, 1, 2))
            gauss0, drop0 = jax.random.split(rng(b["idx"] + 1), 2)
            out = vae_apply(inp, rngs={"gaussian": gauss0, "dropout": drop0}, deterministic=False, method=vae.encode).latent_dist
            return None, ((out.mean, out.std), get_encoded(b["input_ids"], b["attention_mask"]))

        _, (all_vae_out, all_encoded) = lax.scan(_vae_apply, None, batch)
        print(all_vae_out[0].shape, batch["pixel_values"].shape)
        all_encoded = all_encoded.reshape(all_encoded.shape[0], *all_encoded.shape[2:])  # remove batch dim

        def _loss(params, inp):
            itr, (v_mean, v_std), encoded = inp
            sample_rng, noise_rng = jax.random.split(rng(itr), 2)

            latents = jnp.stack([v_mean + v_std * jax.random.normal(r, v_mean.shape) for r in jax.random.split(sample_rng, unet_batch)])
            latents = latents.reshape(unet_batch, context * latents.shape[2], latents.shape[3], latents.shape[4])
            latents = latents.transpose(0, 3, 1, 2)
            latents = latents * 0.18215

            noise = jax.random.normal(noise_rng, latents.shape)
            t0 = jax.random.randint(rng_synced(itr), (unet_batch,), 0, noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(sched_state, latents, noise, t0)

            unet_pred = unet_fn(noisy_latents, encoded, t0, params)
            return distance(unet_pred, noise)

        def _grad(params, inp):
            return jax.value_and_grad(lambda x: _loss(x, inp), has_aux=True)(params)

        def _inner(params):
            def _fn(prev, inp):
                return jax.tree_util.tree_map(operator.add, _grad(params, inp), prev), None

            return _fn

        def _outer(state: TrainState, k):
            ix, av, ae = k
            out = _grad(state.params, (ix[0], (av[0][0], av[1][0]), ae[0]))
            inp = ix[1:], (av[0][1:], av[1][1:]), ae[1:]
            out, _ = lax.scan(_inner(state.params), out, inp)
            scalars, grads = lax.psum(out, "batch")  # we can sum because we divide by device_count^2 above
            return state.apply_gradients(grads=grads), scalars

        def _wrapped(carry, idx):
            ste, av, ae = carry
            ix = batch["idx"].reshape(-1, subsample) + idx * video_group * jax.device_count()
            key = rng_synced(idx + batch["idx"][0])
            av, ae = jax.tree_util.tree_map(lambda x: jax.random.shuffle(key, x).reshape(-1, subsample, *x.shape[1:]), (av, ae))
            ste, sclr = lax.scan(_outer, ste, (ix, av, ae))
            return (ste, av, ae), sclr

        (outer_state, _, _), scalars = lax.scan(_wrapped, (outer_state, all_vae_out, all_encode), jnp.arange(local_iterations))
        return outer_state, (scalars[0].reshape(-1), scalars[1].reshape(-1))

    p_encode_for_sampling = jax.pmap(encode_for_sampling, "batch")
    p_sample = jax.pmap(sample, "batch")
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))

    batch = {"pixel_values": jnp.zeros((jax.local_device_count(), video_group * jax.device_count(), context, resolution, resolution, 3), dtype=jnp.uint8),
             "input_ids": jnp.zeros((jax.local_device_count(), video_group * jax.device_count(), clip_tokens), dtype=jnp.int32),
             "attention_mask": jnp.zeros((jax.local_device_count(), video_group * jax.device_count(), clip_tokens), dtype=jnp.int32),
             "idx": jnp.zeros((jax.local_device_count(),), dtype=jnp.int_)
             }
    compile_fn(lambda: p_train_step(jax_utils.replicate(copy.deepcopy(unet_state)), batch), "train step")
    _, sample_encoded = compile_fn(lambda: p_encode_for_sampling(batch), "sample encoder")
    unet_state = jax_utils.replicate(unet_state)
    compile_fn(lambda: p_sample(unet_state.params, sample_encoded), "sampling")
    del batch, sample_encoded

    def to_img(x: jax.Array) -> wandb.Image:
        return wandb.Image(x.reshape(-1, resolution, 3))

    global_step = 0
    start_time = time.time()
    extra = {}
    lsteps = local_iterations * jax.device_count() // subsample * video_group
    for epoch in range(10 ** 9):
        for i, (vid, ids, msk) in tqdm.tqdm(enumerate(data, 1)):
            global_step += 1
            pid = f'{jax.process_index() * context * jax.local_device_count()}-{(jax.process_index() + 1) * context * jax.local_device_count() - 1}'
            batch = {"pixel_values": vid.astype(jnp.uint8).reshape(jax.local_device_count(), video_group * jax.device_count(), context, resolution, resolution, 3),
                     "input_ids": ids.astype(jnp.int32).reshape(jax.local_device_count(), video_group * jax.device_count(), clip_tokens),
                     "attention_mask": msk.astype(jnp.int32).reshape(jax.local_device_count(), video_group * jax.device_count(), clip_tokens),
                     "idx": jnp.full((jax.local_device_count(),), int(hashlib.blake2b(str(i).encode()).hexdigest()[:4], 16), dtype=jnp.int_)
                     }
            if global_step == 1:
                s_mode, sample_encoded = p_encode_for_sampling(batch)
                extra[f"Samples/Reconstruction (Mode) {pid}"] = to_img(to_host(s_mode, lambda x: x))
            if global_step <= 2:
                log(f"Step {global_step}")
            i *= lsteps

            if i % sample_interval == 0:
                sample_out = p_sample(unet_state.params, sample_encoded)
                for rid, g in enumerate(np.split(to_host(sample_out, lambda x: x), 4, 1)):
                    extra[f"Samples/Reconstruction (U-Net, Guidance {2**rid}) {pid}"] = to_img(g)

            log(f"Before step {i}")
            unet_state, scalars = p_train_step(unet_state, batch)
            log("After")

            timediff = time.time() - start_time
            sclr = to_host(scalars)
            log("To host")

            for offset, (unet_sq, unet_abs) in enumerate(zip(*sclr)):
                vid_per_day = i / timediff * 24 * 3600 * jax.device_count()
                vals = {"U-Net MSE/Total": float(unet_sq), "U-Net MAE/Total": float(unet_abs),
                        "Step": i + offset - lsteps, "Epoch": epoch}
                if offset == lsteps - 1:
                    vals.update({"Runtime": timediff, "Speed/Videos per Day": vid_per_day,
                                 "Speed/Frames per Day": vid_per_day * context})
                    vals.update(extra)
                    extra = {}
                run.log(vals, step=(global_step - 1) * lsteps + offset)
            if i % save_interval == 0 and jax.process_index() == 0:
                for n, s in (("unet", unet_state),):
                    p = to_host(s.params)
                    flattened, jax_structure = jax.tree_util.tree_flatten(p)
                    for _ in range(_UPLOAD_RETRIES):
                        try:
                            with smart_open.open(base_path + n + ".np", "wb") as f:
                                np.savez(f, **{str(i): v for i, v in enumerate(flattened)})
                            break
                        except:
                            log("failed to write", n, "checkpoint")
                            traceback.print_exc()


if __name__ == "__main__":
    app()
