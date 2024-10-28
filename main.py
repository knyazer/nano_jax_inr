import sys

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

from dataloader import load_imagenette_images, prepare_datasets, preprocess_mnist
from evaluation import eval_image
from model import MLP, CombinedModel, LatentMap
from utils import make_mesh

mpl.use("TkAgg")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# jax.config.update("jax_log_compiles", True)

MAX_DIM = 600


class Image(eqx.Module):
    data: Float[Array, "2048 2048 c"]
    shape: Int[Array, "2"]
    channels: int

    def __init__(self, data, shape, channels):
        self.data = data
        self.shape = shape
        self.channels = channels

    def max_shape(self):
        return (MAX_DIM, MAX_DIM)

    def max_latents(self):
        return int(self.max_shape()[0] * self.max_shape()[1] * 0.05)


@jax.profiler.annotate_function
@jax.named_scope("get_latent_points")
def get_latent_points(key: PRNGKeyArray, image: Image, fraction: float = 0.05):
    w, h, c = image.data.shape

    sobel_x = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = jnp.sum(
        jax.vmap(lambda i: convolve(i, sobel_x, mode="same"), in_axes=-1)(image.data), axis=0
    )
    grad_y = jnp.sum(
        jax.vmap(lambda i: convolve(i, sobel_y, mode="same"), in_axes=-1)(image.data), axis=0
    )

    grad_magnitude = jnp.nan_to_num(jnp.sqrt(grad_x**2 + grad_y**2))

    x_grid, y_grid = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing="ij")
    grid = jnp.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)

    num_latents = image.max_latents()
    indices = jr.choice(
        key, grid, shape=(num_latents,), p=grad_magnitude.reshape(-1), replace=False
    )
    num_points = image.shape[0] * image.shape[1] * fraction
    indices = jnp.where(
        (jnp.arange(num_latents) < num_points)[:, None].repeat(2, 1), indices, jnp.nan
    )
    return indices


@jax.profiler.annotate_function
@jax.named_scope("train_step")
def train_step(
    model: CombinedModel,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.nanmean((preds - pixels).ravel() ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


@eqx.filter_jit
@jax.profiler.annotate_function
@jax.named_scope("sample_pixels")
def sample_pixels(
    image: Image,
    key: PRNGKeyArray,
    fraction=0.25,
) -> tuple:
    w, h = image.shape
    W, H = image.max_shape()  # noqa
    indices = jr.choice(key, W * H, shape=(int(W * H * fraction),), replace=False)
    coords = make_mesh((W, H))[indices]
    pixels = image.data[coords[:, 0], coords[:, 1], :]
    return coords, pixels


def train_image(image: Image, key: PRNGKeyArray, epochs: int = 1000) -> CombinedModel:
    """Trains an MLP model to represent a single image."""
    jax.debug.print("entering train image...")
    image_data = image.data
    w, h = image.shape  # 'channels' is separate cuz it must be static
    c = image.channels

    k1, k2, k3, k4 = jr.split(key, 4)

    mlp = MLP(32, [32], c, k1)
    latent_points = get_latent_points(k2, image, fraction=0.05)
    latent_map = LatentMap(k3, latent_points, image)
    model = CombinedModel(image_data, latent_map, mlp)

    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    jax.debug.print("entering scan loop...")

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey, fraction=0.25)
        model, opt_state, loss = train_step(model, optim, opt_state, batch_coords, batch_pixels)
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

    prepare_datasets()

    # Load MNIST data
    logging.info("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist_images, mnist_labels = preprocess_mnist(mnist)
    logging.info("MNIST dataset loaded with %d samples.", mnist_images.shape[0])

    # Load Imagenette images
    logging.info("Loading Imagenette images...")
    imagenette_images = load_imagenette_images()
    logging.info("Imagenette dataset loaded with some (?) samples.")

    # Limit to first 10 images from each dataset for training
    num_train_mnist = 0
    num_train_imagenette = 1

    # Train on MNIST images
    logging.info("Training on first %d MNIST images...", num_train_mnist)
    for idx in range(min(num_train_mnist, mnist_images.shape[0])):
        image = mnist_images[idx].reshape(28, 28)
        key = jr.PRNGKey(idx)
        logging.info("Training MNIST image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=1000)
        psnr = eval_image(model, jnp.array(image), viz="mnist")
        logging.info("PSNR for MNIST image %d: %.2f", idx + 1, psnr)

    # Train on Imagenette images
    logging.info("Training on first %d Imagenette images...", num_train_imagenette)

    imagenette_list = []
    for _ in range(num_train_imagenette):
        image = next(imagenette_images)
        w, h, c = image.shape
        placeholder = jnp.full((MAX_DIM, MAX_DIM, c), jnp.nan)
        placeholder = placeholder.at[:w, :h, :].set(image)
        imagenette_list.append(Image(data=placeholder, shape=jnp.array([w, h]), channels=3))

    imagenette_stacked_data = jnp.stack([image.data for image in imagenette_list])
    imagenette_stacked_shape = jnp.stack([jnp.array(image.shape) for image in imagenette_list])

    imagenette_stacked = Image(
        data=imagenette_stacked_data, shape=imagenette_stacked_shape, channels=3
    )

    key = jr.PRNGKey(42)
    fn = eqx.Partial(train_image, epochs=1000, key=key)
    with jax.profiler.trace("/tmp/tensorboard"):
        models = fn(imagenette_list[0])
    jax.profiler.save_device_memory_profile("memory.prof")
    exit()

    plt.show()


if __name__ == "__main__":
    app.run(main)
