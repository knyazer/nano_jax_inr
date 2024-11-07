import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
from absl import app, flags, logging
from jax_smi import initialise_tracking
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from dataloader import load_imagenette_images, preprocess_mnist
from evaluation import eval_image
from train import train_image
from utils import Image

initialise_tracking()

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "task", None, ["trial", "mnist", "imagenette"], "Type of the task to run.", required=True
)

flags.DEFINE_integer("epochs", 1000, "Number of epochs to train the model.")
flags.DEFINE_integer("num_images", -1, "Number of images to train on, if -1 use all images.")

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("TkAgg")
except Exception as e:
    print(f"Error importing matplotlib, plotting will not work. {e!s}")


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


def data_loader(dataset_name, batch_size, num_devices=1):
    num_images = FLAGS.num_images
    logging.info(f"Loading {num_images if num_images != -1 else 'all'} images...")

    if dataset_name == "mnist":
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        mnist_images, mnist_labels = preprocess_mnist(mnist)
        num_train_images = len(mnist_images) if num_images == -1 else num_images

        # batch image objects
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
            yield image_soa

    elif dataset_name == "imagenette":
        # the strategy is: preload a bs * 20 images, sort the images by one of dims;
        # assign batch sizes as either batch_size or batch_size / 4 based on the size
        def sorted_image_gen():
            cnt = 0
            done = False
            imagenette = load_imagenette_images()
            while (cnt := cnt + 1) < num_images:
                images = []
                for _ in range(10 * batch_size):
                    try:
                        image_data = next(imagenette)
                    except StopIteration:
                        done = True
                        break
                    image_data = jnp.array(image_data)
                    images.append(image_data)
                images = sorted(images, key=lambda x: -x.shape[0] * x.shape[1])

                yield from images
                del images

                if done:
                    break

        def batched_sorted_image_gen():
            # load images, if image is smaller than 1000x1000 batch it with batch_size other images,
            # otherwise batch it with batch_size / 4 other images
            generator = sorted_image_gen()

            while True:
                normal_batch = []
                small_batch = False
                max_h = 0
                max_w = 0
                for _ in range(batch_size):
                    normal_batch.append(next(generator))
                    max_w = max(max_w, normal_batch[-1].shape[0])
                    max_h = max(max_h, normal_batch[-1].shape[1])

                small_batch = max_w > 1000 and max_h > 1000

                # if small_batch:
                #     for chunk_start in range(0, len(normal_batch), batch_size // 4):
                #         yield (
                #             normal_batch[chunk_start : chunk_start + batch_size // 4],
                #             (max_w, max_h),
                #         )
                # else:
                #     yield normal_batch, (max_w, max_h)
                yield normal_batch, (max_w, max_h)
                del normal_batch

        num_images = num_images if num_images != -1 else 100_000_000
        for image_batch, max_sz in batched_sorted_image_gen():
            batch_images = []
            for raw_image_data in image_batch:
                image_data = jnp.array(raw_image_data)
                image = Image(
                    data=image_data,
                    shape=image_data.shape,
                    channels=3,
                    maxsize=max_sz,
                )
                batch_images.append(image)

            stacked_data = jnp.stack([image.data for image in batch_images])
            stacked_shape = jnp.stack([image.shape for image in batch_images])
            del batch_images

            assert stacked_shape.shape[0] == stacked_data.shape[0]

            image_soa = Image(data=stacked_data, shape=stacked_shape, channels=3).shrink_to_grid()
            yield image_soa

    else:
        _msg = f"Dataset {dataset_name} not implemented."
        raise NotImplementedError(_msg)


def bench_dataset(dataset_name):
    total = 0
    batch_size = 2
    if dataset_name == "mnist":
        total = 60000
        batch_size = 64
    elif dataset_name == "imagenette":
        total = 14000  # TODO
        batch_size = 1

    datagen = data_loader(dataset_name, batch_size)

    fn = eqx.filter_jit(eqx.filter_vmap(eqx.Partial(train_image, epochs=FLAGS.epochs)))

    idx = 0
    for image_batch in tqdm(datagen, total=total // batch_size):
        key = jr.key(idx := idx + 1)

        logging.info("Training...")
        model = fn(image_batch, jr.split(key, batch_size))
        logging.info("... done; evaluating...")
        psnr = eqx.filter_vmap(eval_image)(model, image_batch)
        logging.info("... done; Batch PSNR is %.2f +- %.2f", float(psnr.mean()), float(psnr.std()))


def main(argv):
    del argv  # Unused.
    # Configure absl logging
    logging.get_absl_handler().python_handler.stream = sys.stdout
    if jax.process_index() == 0:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.ERROR)

    if FLAGS.task == "trial":
        trial_run()
    elif FLAGS.task == "mnist":
        bench_dataset("mnist")
    elif FLAGS.task == "imagenette":
        bench_dataset("imagenette")
    else:
        _msg = f"Task {FLAGS.task} not implemented. This is... weird. Ask knyazer about it"
        logging.error(_msg)


if __name__ == "__main__":
    app.run(main)
