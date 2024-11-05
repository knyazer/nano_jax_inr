import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from PIL import Image

from model import CombinedModel
from utils import make_mesh


def compute_psnr(
    true_image: Float[Array, "h w *"], pred_image: Float[Array, "h w *"]
) -> Float[Array, ""]:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    true_image = jnp.clip(true_image, 0, 1)
    pred_image = jnp.clip(pred_image, 0, 1)
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
    # rearrange preds according to coords
    pred_image = jnp.zeros((w, h, c))
    pred_image = pred_image.at[coords[:, 0], coords[:, 1]].set(preds)

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
