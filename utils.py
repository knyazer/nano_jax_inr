import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

MAX_DIM = 2048


class Image(eqx.Module):
    data: Float[Array, "* maxd maxd c"]
    shape: Int[Array, "* 2"]
    channels: int
    _max_shape: tuple

    def __init__(self, data, shape, channels, maxsize=None):
        self.data = data
        self.shape = jnp.array(shape)[..., :2]
        self.channels = channels

        if maxsize is not None and maxsize == "auto":
            self._max_shape = (int(self.shape[0]), int(self.shape[1]))
        elif maxsize is None:
            self._max_shape = (MAX_DIM, MAX_DIM)
        else:
            self._max_shape = maxsize

    def max_shape(self):
        return self._max_shape

    def max_latents(self):
        return int(self.max_shape()[0] * self.max_shape()[1] * 0.05)

    def shrink(self):
        with jax.ensure_compile_time_eval():
            # given that we are storing an array of shapes
            assert len(self.shape.shape) == 2
            assert self.shape.shape[0] >= 2
            # we are going to shrink the image to the maximum shape
            new_max_shape = self.shape.max(axis=0)
            new_image = Image(
                self.data, self.shape, self.channels, maxsize=(new_max_shape[0], new_max_shape[1])
            )
            return new_image


def make_mesh(shape):
    return jnp.stack(jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))).reshape(2, -1).T
