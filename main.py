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
from absl import app, flags, logging
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from dataloader import load_imagenette_images, preprocess_mnist
from evaluation import eval_image
from train import train_image
from utils import Image

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "task", None, ["trial", "mnist", "imagenette"], "Type of the task to run.", required=True
)

flags.DEFINE_integer("epochs", 1000, "Number of epochs to train the model.")
flags.DEFINE_integer("num_images", -1, "Number of images to train on, if -1 use all images.")

mpl.use("TkAgg")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def trial_run():
    images = []
    logging.info("Loading trial images..")
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
    logging.info("... done")

    store = [[] for _ in images]
    for key in jr.split(jr.key(0), 10):
        for i, image in enumerate(images):
            model = eqx.filter_jit(train_image)(image, key, epochs=FLAGS.epochs)
            psnr = eval_image(model, image.data)
            logging.info("PSNR for trial image: %.2f", psnr)
            store[i].append(psnr)
    store = jnp.array(store)

    logging.info(f"Mean PSNR for all trial images: {jnp.mean(store, axis=1)}")
    logging.info(f"STD PSNR for all trial images: {jnp.std(store, axis=1)}")


def bench_dataset(dataset_name):
    num_train_images = FLAGS.num_images
    logging.info(f"Loading {num_train_images if num_train_images != -1 else 'all'} images...")

    if dataset_name == "mnist":
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        mnist_images, mnist_labels = preprocess_mnist(mnist)
        batch_size = 128
        num_train_images = num_train_images if num_train_images == -1 else len(mnist_images)

        # batch image objects
        data = []
        for chunk_idx in range(0, num_train_images, batch_size):
            batch_images = []
            for idx in range(chunk_idx, min(chunk_idx + batch_size, num_train_images)):
                image_data = mnist_images[idx].reshape(28, 28)
                image_data = jnp.array(image_data)[..., None]
                image = Image(
                    data=image_data,
                    shape=image_data.shape,
                    channels=1,
                    maxsize=(28, 28),
                )
                batch_images.append(image)

            stacked_data = jnp.stack([image.data for image in batch_images])
            stacked_shape = jnp.stack([image.shape for image in batch_images])

            assert stacked_shape.shape[0] == stacked_data.shape[0]

            image_soa = Image(data=stacked_data, shape=stacked_shape, channels=1, maxsize=(28, 28))
            data.append(image_soa)
    else:
        _msg = f"Dataset {dataset_name} not implemented."
        raise NotImplementedError(_msg)

    logging.info("... done")

    fn = eqx.Partial(train_image, epochs=FLAGS.epochs)

    for i, image in tqdm(enumerate(data), total=len(data)):
        key = jr.key(i)
        model = eqx.filter_vmap(fn)(image, jr.split(key, len(image.shape)))
        psnr = eqx.filter_vmap(eval_image)(model, image)
        logging.info("PSNR for trial image: %.2f +- %.2f", float(psnr.mean()), float(psnr.std()))


def main(argv):
    del argv  # Unused.
    # Configure absl logging
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    if FLAGS.task == "trial":
        trial_run()
    elif FLAGS.task == "mnist":
        bench_dataset("mnist")
    elif FLAGS.task == "imagenette":
        bench_dataset("imagenette")
    else:
        _msg = f"Task {FLAGS.task} not implemented. This is weird... ask knyazer about it"
        logging.error(_msg)


if __name__ == "__main__":
    app.run(main)
