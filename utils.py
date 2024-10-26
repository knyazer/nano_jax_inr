import jax.numpy as jnp


def make_mesh(shape):
    return jnp.stack(jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))).reshape(2, -1).T
