import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from absl import app, logging
from jax.scipy.signal import convolve
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from dataloader import load_imagenette_images, prepare_datasets, preprocess_mnist
from evaluation import eval_image
from model import MLP, CombinedModel, LatentMap
from utils import make_mesh

mpl.use("TkAgg")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

MAX_DIM = 1024


class Image(eqx.Module):
    data: Float[Array, "maxd maxd c"]
    shape: Int[Array, "2"]
    channels: int
    _max_shape: tuple

    def __init__(self, data, shape, channels, maxsize=None):
        self.data = data
        self.shape = shape[:2]
        self.channels = channels

        if maxsize is not None:
            self._max_shape = (int(self.shape[0]), int(self.shape[1]))
        else:
            self._max_shape = (MAX_DIM, MAX_DIM)

    def max_shape(self):
        return self._max_shape

    def max_latents(self):
        return int(self.max_shape()[0] * self.max_shape()[1] * 0.05)


@jax.named_scope("get_latent_points")
def get_latent_points(key: PRNGKeyArray, image: Image, fraction: float = 0.05):
    w, h, c = image.data.shape

    # the following piece of code is somewhat weirdly written:
    # the reason is that we wish to match kornia behaviour closely
    # somehow it is very hard to achieve, but what i did at least matches moments :)
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

    padded_image = jnp.pad(image.data, ((1, 1), (1, 1), (0, 0)), mode="edge")
    grad_x = jax.vmap(lambda i: convolve(i, sobel_x, mode="valid"), in_axes=-1)(padded_image)
    grad_x_sq = jnp.sum(grad_x**2, axis=0)

    grad_y = jax.vmap(lambda i: convolve(i, sobel_y, mode="valid"), in_axes=-1)(padded_image)
    grad_y_sq = jnp.sum(grad_y**2, axis=0)

    grad_magnitude = jnp.nan_to_num(jnp.sqrt(grad_x_sq + grad_y_sq))

    x_grid, y_grid = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing="ij")
    grid = jnp.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)

    num_latents = image.max_latents()
    indices = jr.choice(
        key, grid, shape=(num_latents,), p=grad_magnitude.reshape(-1), replace=False
    )
    num_points = image.shape[0] * image.shape[1] * fraction
    indices = jnp.where((jnp.arange(num_latents) < num_points)[:, None].repeat(2, 1), indices, -1)
    return indices.astype(jnp.int32)


@jax.named_scope("train_step")
def train_step(
    model: CombinedModel,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    # replace nan pixels with m.mu
    no_nan_pixels = jnp.where(jnp.isnan(pixels), model.mu, pixels)
    prod_mask = jnp.where(jnp.isnan(pixels), 0, 1)
    num_good_pixels = jnp.sum(jnp.any(jnp.logical_not(jnp.isnan(pixels)), axis=-1))
    num_good_pixels = jax.lax.cond(num_good_pixels == 0, lambda: 1, lambda: num_good_pixels)

    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.sum(((preds - no_nan_pixels) * prod_mask).ravel() ** 2) / num_good_pixels

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


@jax.named_scope("sample_pixels")
def sample_pixels(
    image: Image,
    key: PRNGKeyArray,
    fraction=0.25,
) -> tuple:
    w, h = image.shape
    W, H = image.max_shape()  # noqa
    indices = jr.choice(
        key, W * H, shape=(int(max(min(W * H * fraction, W * H), 1)),), replace=False
    )
    coords = make_mesh((W, H))[indices]
    pixels = image.data[coords[:, 0], coords[:, 1], :]
    return coords, pixels


@eqx.filter_jit
def train_image(image: Image, key: PRNGKeyArray, epochs: int = 1000) -> CombinedModel:
    """Trains an MLP model to represent a single image."""
    image_data = image.data
    w, h = image.shape  # 'channels' is separate cuz it must be static
    c = image.channels

    k1, k2, k3, k4 = jr.split(key, 4)

    mlp = MLP(32, [32], c, k1)
    latent_points = get_latent_points(k2, image, fraction=0.05)

    latent_map = LatentMap(k3, latent_points, image)
    model = CombinedModel(image_data, latent_map, mlp)
    model = model.check()

    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey, fraction=0.25)
        model, opt_state, loss = train_step(model, optim, opt_state, batch_coords, batch_pixels)
        model = model.check()
        return (model, opt_state, sample_key), loss

    (model, opt_state, _), losses = jax.lax.scan(
        scan_fn, (model, opt_state, k4), None, length=epochs
    )

    return model


def main(argv):
    del argv  # Unused.

    # Configure absl logging
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    images = []
    for file in os.listdir(Path("trial_images")):
        if file.endswith((".png", ".JPEG")):
            image_data = plt.imread(Path("trial_images") / file)
            image_shape = image_data.shape[:2]
            if len(image_data.shape) == 2:
                image_data = image_data[..., None]
                channels = 1
            else:
                channels = 3
            # normalize image data
            if image_data.max() > 2:  # check if we are dealing with 0-255 or 0-1
                image_data = jnp.array(image_data, dtype=jnp.float32) / 255.0
            images.append(
                Image(
                    data=jnp.array(image_data),
                    shape=jnp.array(image_shape),
                    channels=channels,
                    maxsize="auto",
                )
            )

    fn = eqx.Partial(train_image, epochs=1000)

    store = [[] for _ in images]
    for key in jr.split(jr.key(0), 10):
        for i, image in enumerate(images):
            model = fn(image, key)
            psnr = eval_image(model, image.data)
            logging.info("PSNR for trial image: %.2f", psnr)
            store[i].append(psnr)

    store = jnp.array(store)

    logging.info(f"Mean PSNR for all trial images: {jnp.mean(store, axis=1)}")
    logging.info(f"STD PSNR for all trial images: {jnp.std(store, axis=1)}")


if __name__ == "__main__":
    app.run(main)
