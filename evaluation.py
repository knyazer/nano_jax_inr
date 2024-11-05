import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from PIL import Image

from model import CombinedModel
from utils import Image, make_mesh


def compute_psnr(true_image: Float[Array, "*"], pred_image: Float[Array, "*"]) -> Float[Array, ""]:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    true_image = jnp.clip(true_image, 0, 1)
    pred_image = jnp.clip(pred_image, 0, 1)
    mse = jnp.nanmean((true_image - pred_image) ** 2)
    max_pixel = 1.0
    psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse))
    return psnr


@eqx.filter_jit
def eval_image(model: CombinedModel, image: Image) -> Float[Array, ""]:
    w, h = image.max_shape()
    coords = make_mesh((w, h))

    preds = eqx.filter_vmap(model)(coords)
    ground_truth = image.data[coords[:, 0], coords[:, 1]]

    psnr = compute_psnr(preds, ground_truth)
    return psnr
