import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
from absl import app, logging

from evaluation import eval_image
from train import train_image
from utils import Image

mpl.use("TkAgg")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

MAX_DIM = 1024


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
