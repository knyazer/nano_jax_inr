import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from absl import app, logging
from jax.scipy.signal import convolve
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from PIL import Image
from sklearn.datasets import fetch_openml

from dataloader import load_imagenette_images, prepare_datasets, preprocess_mnist

mpl.use("TkAgg")


def make_mesh(shape):
    return jnp.stack(jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))).reshape(2, -1).T


class LatentMap(eqx.Module):
    """Parametrized function to convert point in 2D (pixel coordinate) into a latent."""

    positions: Int[Array, "n dim"]
    neighbor_map: Int[Array, "w h num_neighbors"]
    embeddings: Array

    def __init__(self, key, positions, shape=(28, 28)):
        positions = jnp.array(positions)

        def compute_nearest_indices(request_pos):
            distances = jnp.linalg.norm(jnp.array(positions) - request_pos, axis=1)
            return jnp.argsort(distances)[:4]

        coords = make_mesh(shape)
        nearest_indices = jax.vmap(compute_nearest_indices)(coords)
        neighbor_map = jnp.zeros((shape[0], shape[1], 4))
        neighbor_map = neighbor_map.at[coords[:, 0], coords[:, 1]].set(nearest_indices)

        self.neighbor_map = neighbor_map.astype(jnp.int16)
        self.positions = positions
        self.embeddings = jr.normal(key, (len(positions), 32)) * 0.1

    def __call__(self, position: Float[Array, "2"], *, ensure_valid=True):
        int_pos = jnp.floor(position).astype(jnp.int32)
        # linear-ish interpolation of neighbors
        if ensure_valid:
            int_pos = eqx.error_if(
                int_pos,
                jnp.any(
                    jnp.logical_or(int_pos < 0, int_pos >= jnp.array(self.neighbor_map.shape[:2]))
                ),
                f"Requested position is out of bounds: {self.neighbor_map.shape[:2]}",
            )

        neighbors = self.neighbor_map[int_pos[0], int_pos[1]]
        distances = jnp.linalg.norm(self.positions[neighbors] - int_pos, axis=1)
        weights = 1.0 / (distances + 1e-8)

        latent = jnp.sum(weights[:, None] * self.embeddings[neighbors], axis=0) / jnp.sum(weights)
        return jax.lax.cond(weights.sum() > 1e3, lambda: latent, lambda: jnp.zeros_like(latent))


"""
latent_map = LatentMap(jr.PRNGKey(0), [(1, 1), (3, 3), (1, 3), (3, 1), (0, 0)], (4, 4))
assert jnp.allclose(latent_map(jnp.array([0, 0], dtype=jnp.int32)), latent_map.embeddings[4])
assert jnp.allclose(latent_map(jnp.array([1, 1], dtype=jnp.int32)), latent_map.embeddings[0])
assert jnp.allclose(
    latent_map(jnp.array([2, 2], dtype=jnp.int32)),
    jnp.mean(latent_map.embeddings[:-1], axis=0),
)
del latent_map
"""


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, key: PRNGKeyArray):
        keys = jr.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))

    def __call__(self, x: Float[Array, "latent_dim"]) -> Float[Array, "..."]:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return jnp.clip(x, 0, 1)


class CombinedModel(eqx.Module):
    """Needed to simplify the logic, and in case you desire to have some glue."""

    latent_map: LatentMap
    mlp: MLP

    def __call__(self, x: Int[Array, "2"]) -> Float[Array, "..."]:
        latent = self.latent_map(x)
        return self.mlp(latent)


def compute_psnr(
    true_image: Float[Array, "h w *"], pred_image: Float[Array, "h w *"]
) -> Float[Array, ""]:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = jnp.mean((true_image - pred_image) ** 2)
    max_pixel = 1.0
    psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse))
    return psnr


def eval_image(
    model: CombinedModel, image: Float[Array, "w h *"], viz: str | None = None
) -> Float[Array, ""]:
    if len(image.shape) == 2:
        image = image[..., None]
    w, h, c = image.shape[:3]

    coords = make_mesh((w, h))
    preds = eqx.filter_vmap(model)(coords)
    pred_image = preds.reshape(w, h) if c == 1 else preds.reshape(w, h, c)

    image = image.squeeze()
    pred_image = pred_image.squeeze()

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
        plt.figure()
        plt.imshow(combined_image)
        with path.open("wb") as f:
            combined_image.save(f)
            time.sleep(0.05)
    return psnr


def get_latent_points(key: PRNGKeyArray, image: Float[Array, "w h c"], fraction: float = 0.05):
    w, h, c = image.shape

    sobel_x = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = jnp.mean(
        jax.vmap(lambda i: convolve(i, sobel_x, mode="same"), in_axes=-1)(image), axis=0
    )
    grad_y = jnp.mean(
        jax.vmap(lambda i: convolve(i, sobel_y, mode="same"), in_axes=-1)(image), axis=0
    )

    grad_magnitude = jnp.sqrt(grad_x**2 + grad_y**2)
    probs = grad_magnitude.flatten() / jnp.sum(grad_magnitude)
    num_points = max(1, int(w * h * fraction))

    indices = jr.choice(key, w * h, shape=(num_points,), p=probs)

    coords = make_mesh((w, h))
    return coords[indices]


def train_step(
    model: CombinedModel,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.mean((preds.ravel() - pixels.ravel()) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


def sample_pixels(image: Float[Array, "w h *"], key: PRNGKeyArray, fraction: float = 0.25) -> tuple:
    w, h = image.shape[:2]
    num = max(1, int(w * h * fraction))
    indices = jr.choice(key, w * h, shape=(num,), replace=False)
    coords = make_mesh((w, h))[indices]
    pixels = image[coords[:, 1], coords[:, 0], :]
    return coords, pixels


def train_image(
    image: Float[Array, "w h *"], key: PRNGKeyArray, epochs: int = 1000
) -> CombinedModel:
    """Trains an MLP model to represent a single image."""
    if len(image.shape) == 2:
        image = image[..., None]
    w, h, c = image.shape

    k1, k2, k3 = jr.split(key, 3)
    mlp = MLP(32, [128], c, k1)

    latent_points = get_latent_points(k2, image, fraction=0.05)
    latent_map = LatentMap(k3, latent_points, image.shape)
    model = CombinedModel(latent_map, mlp)

    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey, fraction=0.25)
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
        continue
        image = mnist_images[idx].reshape(28, 28)
        key = jr.PRNGKey(idx + 1)
        logging.info("Training MNIST image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=10000)
        psnr = eval_image(model, jnp.array(image), viz="mnist")
        logging.info("PSNR for MNIST image %d: %.2f", idx + 1, psnr)
        plt.show()

    # Train on Imagenette images
    logging.info("Training on first %d Imagenette images...", num_train_imagenette)
    for idx, image in enumerate(imagenette_images):
        key = jr.PRNGKey(idx + 100)
        logging.info("Training Imagenette image %d", idx + 1)
        model = train_image(jnp.array(image), key, epochs=1)
        psnr = eval_image(model, jnp.array(image), viz="imagenet")
        logging.info("PSNR for Imagenette image %d: %.2f", idx + 1, psnr)
        if idx == num_train_imagenette - 1:
            break


if __name__ == "__main__":
    app.run(main)
