from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from PIL import Image

from config import get_config as C
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
def eval_image(model: CombinedModel, image: Image):
    w, h = image.max_shape()
    coords = make_mesh((w, h))

    preds = jnp.clip(eqx.filter_vmap(model)(coords), 0, 1)
    ground_truth = image.data[coords[:, 0], coords[:, 1]]
    psnr = compute_psnr(preds, ground_truth)

    return psnr, preds.reshape((w, h, C().out_dim))


def record_results(model_chks, preds, path: Path):
    # path is the path of the image, so we want to remove file extension and rename the root folder
    with jax.ensure_compile_time_eval():
        path.mkdir(parents=True)
        for idx, model in model_chks.items():
            latents = model.latent_map.embeddings
            latents = latents[~jnp.isnan(latents).any(axis=1)]
            jnp.save(path / Path(f"latents_{idx}.npy"), latents)
        if preds.shape[-1] == 1:
            preds = preds[..., 0]
        if len(preds.shape) == 2:
            plt.imsave(path / Path("preds.png"), preds.T)
        else:
            plt.imsave(path / Path("preds.png"), jnp.swapaxes(preds, 0, 1))
