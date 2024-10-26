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
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from sklearn.datasets import fetch_openml

from dataloader import load_imagenette_images, prepare_datasets, preprocess_mnist
from evaluation import eval_image
from model import MLP, CombinedModel, LatentMap
from utils import make_mesh

mpl.use("TkAgg")


def get_latent_points(key: PRNGKeyArray, image: Float[Array, "w h c"], fraction: float = 0.05):
    w, h, c = image.shape

    sobel_x = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = jnp.sum(
        jax.vmap(lambda i: convolve(i, sobel_x, mode="same"), in_axes=-1)(image), axis=0
    )
    grad_y = jnp.sum(
        jax.vmap(lambda i: convolve(i, sobel_y, mode="same"), in_axes=-1)(image), axis=0
    )

    grad_magnitude = jnp.sqrt(grad_x**2 + grad_y**2)
    plt.figure()
    plt.imshow(grad_magnitude)

    num_points = max(1, int(w * h * fraction))
    x_grid, y_grid = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing="ij")
    grid = jnp.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)

    indices = jr.choice(key, grid, shape=(num_points,), p=grad_magnitude.reshape(-1), replace=False)
    return indices


def train_step(
    model: CombinedModel,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.mean((preds - pixels).ravel() ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


def sample_pixels(image: Float[Array, "w h *"], key: PRNGKeyArray, fraction: float = 0.25) -> tuple:
    w, h = image.shape[:2]
    num = max(1, int(w * h * fraction))
    indices = jr.choice(key, w * h, shape=(num,), replace=False)
    coords = make_mesh((w, h))[indices]
    pixels = image[coords[:, 0], coords[:, 1], :]
    return coords, pixels.reshape(-1, image.shape[-1])


def train_image(
    image: Float[Array, "w h *"], key: PRNGKeyArray, epochs: int = 1000
) -> CombinedModel:
    """Trains an MLP model to represent a single image."""
    if len(image.shape) == 2:
        image = image[..., None]
    w, h, c = image.shape

    k1, k2, k3, k4 = jr.split(key, 4)

    mlp = MLP(32, [32], c, k1)
    latent_points = get_latent_points(k2, image, fraction=0.05)
    latent_map = LatentMap(k3, latent_points, image.shape)
    model = CombinedModel(image, latent_map, mlp)

    latent_points_viz = jnp.zeros((w, h, c))
    latent_points_viz = latent_points_viz.at[latent_points[:, 0], latent_points[:, 1], 0].set(1)
    plt.figure()
    plt.imshow(latent_points_viz)

    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey, fraction=0.25)
        model, opt_state, loss = train_step(model, optim, opt_state, batch_coords, batch_pixels)
        return (model, opt_state, sample_key), loss

    for i in range(10):
        (model, opt_state, _), losses = jax.lax.scan(
            scan_fn, (model, opt_state, k4), None, length=epochs // 10
        )
        if i != 9:
            psnr = eval_image(model, jnp.array(image))
            logging.info("PSNR during training: %.2f", psnr)

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
    num_train_mnist = 1
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
    imagenette_list = [next(imagenette_images) for _ in range(num_train_imagenette)]
    for idx, image in enumerate(imagenette_list):
        key = jr.PRNGKey(idx + 100)
        logging.info("Training Imagenette image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=1000)
        psnr = eval_image(model, jnp.array(image), viz="imagenet")
        logging.info("PSNR for Imagenette image %d: %.2f", idx + 1, psnr)
        jax.profiler.save_device_memory_profile("memory.prof")
        if idx == num_train_imagenette - 1:
            break
    plt.show()


if __name__ == "__main__":
    app.run(main)
