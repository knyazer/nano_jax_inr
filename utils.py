import math

import equinox as eqx
import jax
import jax.numpy as jnp
from absl import logging
from jaxtyping import Array, Float, Int

MAX_DIM = 5000


# Grid is a list of allowed sizes of images: ie [(32, 32), (50, 50)] etc etc
# The point is to reduce compilation time, while simultaneously allowing for
# use of batching on the images, if they are small
_grid_size_distr = [
    150,
    224,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    700,
    800,
    1000,
    1500,
    2000,
]


def _generate_grid():
    g = []
    for w in _grid_size_distr:
        for h in _grid_size_distr:
            if w == h:  # w / h < 2.5 and h / w < 2.5:
                g.append((w, h))
    g.append((MAX_DIM, MAX_DIM))
    return g


_GRID = _generate_grid()


class Image(eqx.Module):
    data: Float[Array, "* maxd maxd c"]
    shape: Int[Array, "* 2"]
    channels: int
    _max_shape: tuple

    @staticmethod
    def fake_stacked_grid_generator(batch_size: int, channels=3):
        shape = jnp.zeros((batch_size, 2)).astype(jnp.int32)
        for maxsize in _GRID:
            data_shape = (batch_size, *maxsize, channels)
            yield Image(jnp.zeros(data_shape), shape, channels, maxsize=maxsize)

    def __init__(self, data, shape, channels, maxsize: str | tuple = "auto"):
        self.shape = jnp.array(shape)[..., :2]
        self.channels = channels

        # figure out max shape
        if maxsize == "auto":
            if self.shape.size == 2 and self.shape.ndim == 1:
                self._max_shape = (int(self.shape[0]), int(self.shape[1]))
            else:
                # max size is the maximum of the shape
                max_shape = self.shape.max(axis=0)
                self._max_shape = (int(max_shape[0]), int(max_shape[1]))
        elif maxsize is None:
            self._max_shape = (MAX_DIM, MAX_DIM)
        else:
            self._max_shape = maxsize  # type: ignore

        # figure out whether we need to pad the image data
        if self.shape.size == 2 and self.shape.ndim == 1:
            placeholder = jnp.full((*self.max_shape(), channels), jnp.nan)
            w, h = self.shape
            placeholder = placeholder.at[:w, :h, :].set(data)
            self.data = placeholder
        else:
            placeholder = jnp.full((data.shape[0], *self.max_shape(), channels), jnp.nan)
            w = min(data.shape[-3], self.max_shape()[0])
            h = min(data.shape[-2], self.max_shape()[1])
            placeholder = placeholder.at[:, :w, :h].set(data[:, :w, :h])
            self.data = placeholder[..., : self.max_shape()[0], : self.max_shape()[1], :]

    def max_shape(self):
        return self._max_shape

    def max_latents(self):
        return int(self.max_shape()[0] * self.max_shape()[1] * 0.05)

    def shrink(self):
        with jax.ensure_compile_time_eval():
            # given that we are storing an array of shapes
            assert len(self.shape.shape) == 2
            # we are going to shrink the image to the maximum shape
            new_max_shape = self.shape.max(axis=0)
            new_image = Image(
                self.data,
                self.shape,
                self.channels,
                maxsize=(int(new_max_shape[0]), int(new_max_shape[1])),
            )
            return new_image

    def shrink_to_grid(self):
        # shrinks to the smallest grid size that is larger than max size
        with jax.ensure_compile_time_eval():
            assert len(self.shape.shape) == 2
            new_max_shape = self.shape.max(axis=0)
            new_max_shape = (int(new_max_shape[0]), int(new_max_shape[1]))
            original_shape = new_max_shape

            min_grid_area = 1e10
            for grid_w, grid_h in _GRID:
                if (
                    grid_w >= new_max_shape[0]
                    and grid_h >= new_max_shape[1]
                    and grid_w * grid_h < min_grid_area
                ):
                    min_grid_area = grid_w * grid_h
                    new_max_shape = (grid_w, grid_h)

            logging.info(f"Expanded from {original_shape} to {new_max_shape}")
            new_image = Image(
                self.data,
                self.shape,
                self.channels,
                maxsize=new_max_shape,
            )

            return new_image


def make_mesh(shape):
    return jnp.stack(jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))).reshape(2, -1).T
