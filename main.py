import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from absl import app, logging
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from PIL import Image
from sklearn.datasets import fetch_openml

from dataloader import load_imagenette_images, prepare_datasets, preprocess_mnist


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        key: PRNGKeyArray,
        scale: float = 1e-2,
    ):
        keys = jr.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.scale = scale
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, "..."]:
        x = x * self.scale
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return jnp.clip(x, 0, 1)


@eqx.filter_jit
def train_step(
    model: MLP,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    """Performs a single training step."""

    def loss_fn(m: MLP) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.mean((preds - pixels) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


def sample_pixels(image: Float[Array, "h w *"], key: PRNGKeyArray, fraction: float = 0.25) -> tuple:
    height, width = image.shape[:2]
    num = max(1, int(height * width * fraction))
    indices = jr.choice(key, height * width, shape=(num,), replace=False)
    coords = jnp.stack(jnp.unravel_index(indices, (height, width)), axis=-1)
    if image.ndim == 2:
        pixels = image[coords[:, 1], coords[:, 0]].reshape(-1, 1)
    else:
        pixels = image[coords[:, 1], coords[:, 0], :]
    return coords, pixels


def compute_psnr(
    true_image: Float[Array, "h w *"], pred_image: Float[Array, "h w *"]
) -> Float[Array, ""]:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = jnp.mean((true_image - pred_image) ** 2)
    max_pixel = 1.0
    psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse))
    return psnr


def eval_image(
    model: MLP, image: Float[Array, "h w *"], viz: str | None = None
) -> Float[Array, ""]:
    """Generates the predicted image and computes PSNR."""

    if image.ndim == 2:
        height, width = image.shape
        out_dim = 1
    else:
        height, width, channels = image.shape
        out_dim = channels

    coords = jnp.stack(jnp.meshgrid(jnp.arange(width), jnp.arange(height)), axis=-1).reshape(-1, 2)
    preds = eqx.filter_vmap(model)(coords)
    if out_dim == 1:
        pred_image = preds.reshape(height, width)
    else:
        pred_image = preds.reshape(height, width, out_dim)

    psnr = eqx.filter_jit(compute_psnr)(image, pred_image)

    if viz is not None:
        import numpy  # noqa

        Path("log").mkdir(parents=True, exist_ok=True)
        true_image_pil = Image.fromarray(numpy.array(image * 255).astype(numpy.uint8))
        pred_image_pil = Image.fromarray(numpy.array(pred_image * 255).astype(numpy.uint8))

        index = 0
        while (path := Path(f"log/{viz}_{index}.png")).exists():
            index += 1
            if index > 100:
                raise RuntimeError("Too many images saved: 100, probably wrong.")

        combined_image = Image.new(
            "RGB", (true_image_pil.width + pred_image_pil.width, true_image_pil.height)
        )
        combined_image.paste(true_image_pil, (0, 0))
        combined_image.paste(pred_image_pil, (true_image_pil.width, 0))
        with path.open("wb") as f:
            combined_image.save(f)
            time.sleep(0.05)
    return psnr


def train_image(image: Float[Array, "w h *"], key: PRNGKeyArray, epochs: int = 1000) -> MLP:
    """Trains an MLP model to represent a single image."""
    if image.ndim == 2:
        height, width = image.shape
        out_dim = 1
    else:
        height, width, channels = image.shape
        out_dim = channels

    k1, k2 = jr.split(key)
    model = MLP(2, [64, 64], out_dim, k1, scale=1.0 / float(max(width, height)))
    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey)
        model, opt_state, loss = train_step(model, optim, opt_state, batch_coords, batch_pixels)
        return (model, opt_state, sample_key), loss

    (model, opt_state, _), losses = jax.lax.scan(
        scan_fn, (model, opt_state, k2), None, length=epochs
    )

    return model


def main(argv):
    del argv  # Unused.

    # Configure absl logging
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    # Prepare datasets
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
    num_train_mnist = 3
    num_train_imagenette = 3

    # Train on MNIST images
    logging.info("Training on first %d MNIST images...", num_train_mnist)
    for idx in range(min(num_train_mnist, mnist_images.shape[0])):
        image = mnist_images[idx].reshape(28, 28)
        key = jr.PRNGKey(idx)
        logging.info("Training MNIST image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=10000)
        psnr = eval_image(model, jnp.array(image), viz="mnist")
        logging.info("PSNR for MNIST image %d: %.2f", idx + 1, psnr)

    # Train on Imagenette images
    logging.info("Training on first %d Imagenette images...", num_train_imagenette)
    for idx, image in enumerate(imagenette_images):
        key = jr.PRNGKey(idx + 100)
        logging.info("Training Imagenette image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=1000)
        psnr = eval_image(model, jnp.array(image), viz="imagenet")
        logging.info("PSNR for Imagenette image %d: %.2f", idx + 1, psnr)
        if idx == num_train_imagenette - 1:
            break


if __name__ == "__main__":
    app.run(main)
