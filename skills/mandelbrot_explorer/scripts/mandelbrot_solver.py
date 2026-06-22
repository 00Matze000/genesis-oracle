import jax
import jax.numpy as jnp

@jax.jit
def mandelbrot_kernel(c, max_iters):
    def body_fn(val):
        z, count, active = val
        next_z = z**2 + c
        next_active = active & (jnp.abs(next_z) <= 2.0)
        next_count = jnp.where(next_active, count + 1, count)
        return next_z, next_count, next_active

    def cond_fn(val):
        _, count, active = val
        return jnp.any(active) & (jnp.min(count) < max_iters)

    z = jnp.zeros_like(c)
    count = jnp.zeros_like(c, dtype=jnp.int32)
    active = jnp.ones_like(c, dtype=jnp.bool_)

    _, final_counts, _ = jax.lax.while_loop(cond_fn, body_fn, (z, count, active))
    return final_counts
