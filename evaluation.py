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
        path.mkdir(parents=True, exist_ok=True)
        last_key = list(model_chks.keys())[-1]
        mask = jnp.isnan(model_chks[last_key].latent_map.embeddings).any(axis=1)
        jnp.save(path / Path("positions.npy"), model_chks[last_key].latent_map.positions[~mask])
        for idx, model in model_chks.items():
            latents = model.latent_map.embeddings
            jnp.save(path / Path(f"latents_{idx}.npy"), latents[~mask])
            w_preds = preds[idx]
            if w_preds.shape[-1] == 1:
                w_preds = w_preds[..., 0]
            w_preds = w_preds.T if len(w_preds.shape) == 2 else jnp.swapaxes(w_preds, 0, 1)
            plt.imsave(path / Path(f"preds_{idx}.png"), w_preds)
