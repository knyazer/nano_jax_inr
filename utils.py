from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from absl import logging
from jaxtyping import Array, Float, Int

from config import get_config as C  # noqa

MAX_DIM = 5000


# Grid is a list of allowed sizes of images: ie [(32, 32), (50, 50)] etc etc
# The point is to reduce compilation time, while simultaneously allowing for
# use of batching on the images, if they are small
_grid_size_distr = [
    200,
    225,
    250,
    275,
    300,
    325,
    350,
    375,
    400,
    450,
    475,
    500,
    525,
    550,
    600,
    700,
    800,
    1000,
    1300,
    1500,
    2048,
]


def _generate_grid():
    g = []
    for w in _grid_size_distr:
        for h in _grid_size_distr:
            g.append((w, h))
    g.append((MAX_DIM, MAX_DIM))
    return g


_GRID = _generate_grid()


class Image(eqx.Module):
    channels: int = eqx.field(static=True)
    _max_shape: tuple = eqx.field(static=True)
    data: Float[Array, "* maxd maxd c"]
    shape: Int[Array, "* 2"]

    def __init__(self, data, shape, channels, maxsize: str | tuple = "auto"):
        self.shape = jnp.array(shape)[..., :2]
        self.channels = int(channels)

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
        if type(C().num_latents) is float:
            return int(self.max_shape()[0] * self.max_shape()[1] * C().num_latents)
        return C().num_latents

    def enlarge(self, new_max_shape):
        new_max_shape = (int(new_max_shape[0]), int(new_max_shape[1]))
        new_image = Image(
            self.data,
            self.shape,
            self.channels,
            maxsize=new_max_shape,
        )
        return new_image

    def shrink(self):
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


def make_target_path(path):
    return Path("Processed Images", C().alias, *path.parts[1:-1], Path(path.parts[-1]).stem)
