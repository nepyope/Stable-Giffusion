
def scale_by_laprop(b1: float, b2: float, eps: float, lr: optax.Schedule, clip: float = 1e-2) -> GradientTransformation: #LION Optimizer
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
            return jnp.sign(update * -lr(count)), (mom + delta * (1 - b2)).astype(dtype)

        leaves, treedef = jax.tree_util.tree_flatten(updates)
        all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in (params, state["momentum"])]
        updates, mom = [treedef.unflatten(leaf) for leaf in zip(*[get_update(*xs) for xs in zip(*all_leaves)])]

        return updates, {"momentum": mom, "count": count}

    return GradientTransformation(init_fn, update_fn)