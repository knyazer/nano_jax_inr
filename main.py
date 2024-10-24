from typing import Any, Generator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from PIL import Image
from sklearn.datasets import fetch_openml


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, key: PRNGKeyArray):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


@eqx.filter_jit
def train_step(
    model: MLP,
    optimizer: PyTree,
    coords: Float[Array, "..."],
    pixels: Float[Array, "..."],
) -> tuple:
    """Performs a single training step."""

    def loss_fn(m: MLP) -> Float[Array, ""]:
        preds = m(coords)
        return jnp.mean((preds - pixels) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, optimizer.state)
    new_model = eqx.apply_updates(model, updates)
    new_optimizer = optimizer.replace(state=opt_state)
    return new_model, new_optimizer, loss


@eqx.filter_vmap
def sample_pixels(
    coords: Float[jnp.ndarray, "pixels 2"],
    pixels: Float[jnp.ndarray, "..."],
    key: Array[jnp.int32, "key"],
) -> Tuple[Float[jnp.ndarray, "batch 2"], Float[jnp.ndarray, "..."]]:
    """Randomly samples 25% of the pixels."""
    num = max(1, coords.shape[0] // 4)
    indices = jax.random.choice(key, coords.shape[0], shape=(num,), replace=False)
    return coords[indices], pixels[indices]


def get_coords(width: int, height: int) -> Float[jnp.ndarray, "pixels 2"]:
    """Generates a normalized coordinate grid for the image."""
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    grid = jnp.stack(jnp.meshgrid(x, y, indexing="xy"), -1).reshape(-1, 2)
    return grid


def preprocess_mnist(mnist_data: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses MNIST data."""
    X = mnist_data.data.astype(np.float32) / 255.0  # Normalize to [0,1]
    y = mnist_data.target.astype(np.int32)
    return X, y


def load_imagenette_images() -> Generator[np.ndarray, None, None]:
    """Loads Imagenette images as numpy arrays."""
    for class_dir in IMAGENETTE_DIR.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert("RGB")
                            img = img.resize((64, 64))
                            img_array = (
                                np.array(img).astype(np.float32) / 255.0
                            )  # Normalize to [0,1]
                            yield img_array
                    except Exception as e:
                        logging.warning("Failed to load image %s: %s", img_path, e)


def train_image(image: jnp.ndarray, key: Array[jnp.int32, "key"], epochs: int = 1000) -> MLP:
    """Trains an MLP model to represent a single image."""
    if image.ndim == 2:
        height, width = image.shape
        out_dim = 1
        pixels = image.flatten().reshape(-1, 1)
    else:
        height, width, channels = image.shape
        out_dim = channels
        pixels = image.reshape(-1, channels)

    coords = get_coords(width, height)
    model = MLP(2, [256, 256, 256], out_dim, key)
    optimizer = Adam(model, lr=1e-3)
    sample_key = key

    for epoch in range(1, epochs + 1):
        sample_key, subkey = jax.random.split(sample_key)
        batch_coords, batch_pixels = sample_pixels(coords, pixels, subkey)
        model, optimizer, loss = train_step(model, optimizer, batch_coords, batch_pixels)
        if epoch == 1 or epoch % 100 == 0:
            logging.info("Epoch %d, Loss: %.6f", epoch, loss)
    return model


def main(argv):
    del argv  # Unused.

    # Configure absl logging
    logging.get_absl_handler().use_absl_log_file("implicit_image_representation", "log")
    logging.set_verbosity(logging.INFO)

    # Prepare datasets
    prepare_datasets()

    # Load MNIST data
    logging.info("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_mnist, y_mnist = preprocess_mnist(mnist)
    logging.info("MNIST dataset loaded with %d samples.", X_mnist.shape[0])

    # Load Imagenette images
    logging.info("Loading Imagenette images...")
    imagenette_images = list(load_imagenette_images())
    logging.info("Imagenette dataset loaded with %d images.", len(imagenette_images))

    # Limit to first 10 images from each dataset for training
    num_train_mnist = 10
    num_train_imagenette = 10

    # Train on MNIST images
    logging.info("Training on first %d MNIST images...", num_train_mnist)
    for idx in range(min(num_train_mnist, X_mnist.shape[0])):
        image = X_mnist[idx].reshape(28, 28)
        key = jax.random.PRNGKey(idx)
        logging.info("Training MNIST image %d", idx + 1)
        train_image(jnp.array(image), key, epochs=1000)

    # Train on Imagenette images
    logging.info("Training on first %d Imagenette images...", num_train_imagenette)
    for idx, image in enumerate(imagenette_images[:num_train_imagenette]):
        key = jax.random.PRNGKey(idx + 100)
        logging.info("Training Imagenette image %d", idx + 1)
        train_image(jnp.array(image), key, epochs=1000)


if __name__ == "__main__":
    app.run(main)
