import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import jax.sharding as jshard
from absl import app, flags, logging
from jax.experimental import mesh_utils
from jax_smi import initialise_tracking
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from config import get_config as C  # noqa
from config import load_config_from_py
from dataloader import load_imagenette_images, preprocess_mnist
from evaluation import eval_image, record_results
from train import train_decoder, train_image
from utils import Image, make_target_path

initialise_tracking()

try:
    jax.distributed.initialize()
except Exception:
    print("No distributed training possible: are you multi-gpu?")

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "task",
    None,
    ["trial", "mnist", "imagenette", "ablation"],
    "Type of the task to run. Ablation traverses a given folder.",
    required=True,
)
flags.DEFINE_string(
    "config", None, "Path to the config file. It is in the root of the folder with images."
)
flags.DEFINE_bool("overwrite", False, "Overwrite the results if they already exist.")

flags.DEFINE_bool(
    "use_grid",
    False,  # noqa
    "Use a set of predetermined placheloder sizes when training for images."
    "DO NOT ENABLE IF IMAGES ARE SAME SIZE. Also, feel free to modify the _GRID list in utils.py",
)
flags.DEFINE_bool(
    "use_cached_decoder",
    False,  # noqa
    "Use a trained decoder from the last run: disable for real experiments!!!",
)

# general
flags.DEFINE_integer("num_images", -1, "Number of images to train on, if -1 use all images.")
flags.DEFINE_boolean(
    "ignore_big",
    False,  # noqa
    "Ignore big (1500x1500+) images. Improves performance wastly.",
)
flags.DEFINE_integer(
    "batch_size", 1, "Per-device batch size, increase until OOM or performance stagnates."
)

# some debugging-ish flags
flags.DEFINE_enum(
    "on_error",
    "skip",
    ["skip", "stop", "attempt"],
    "What to do on an error. Useful for debugging, bad otherwise: "
    "the errors could be something like 'not enough pixels to sample',"
    "which probably should be just ignored (ie skipped)",
)

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("TkAgg")
except Exception as e:
    print(f"Error importing matplotlib, plotting will not work. {e!s}.\nThis is not a big deal.")


jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def trial_run():
    images = []
    logging.info("Loading trial images..")
    for file in os.listdir(Path("trial_images")):
        if file.endswith((".png", ".JPEG")):
            image_data = plt.imread(Path("trial_images") / file)  # type: ignore
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
            models = eqx.filter_jit(train_image)(image, key, epochs_list=[FLAGS.epochs])
            psnr = eval_image(models[-1], image.data)
            logging.info("PSNR for trial image: %.2f", psnr)
            store[i].append(psnr)
    store = jnp.array(store)

    logging.info(f"Mean PSNR for all trial images: {jnp.mean(store, axis=1)}")
    logging.info(f"STD PSNR for all trial images: {jnp.std(store, axis=1)}")


SKIPPED = 0


def data_loader(dataset_name):  # noqa
    num_images = FLAGS.num_images
    logging.info(f"Loading {num_images if num_images != -1 else 'all'} images...")
    num_images = 100_000_000 if num_images == -1 else num_images

    if dataset_name == "mnist":
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        mnist_images, mnist_labels = preprocess_mnist(mnist)

        # batch image objects
        for idx in range(num_images):
            image_data = mnist_images[idx].reshape(28, 28)
            image_data = jnp.array(image_data)[..., None]
            image = Image(
                data=image_data,
                shape=image_data.shape,
                channels=1,
                maxsize=(28, 28),
            )
            yield image, Path(f"mnist_cache/{idx}.png")

    elif dataset_name == "imagenette":
        # the strategy is: preload a bs * 20 images, sort the images by one of dims;
        # assign batch sizes as either batch_size or batch_size / 4 based on the size
        def sorted_image_gen():
            cnt = 0
            done = False
            imagenette = load_imagenette_images()
            while True:
                images = []
                for _ in range(100):
                    if (cnt := cnt + 1) > num_images:
                        done = True
                        break
                    try:
                        image_data, image_s = next(imagenette)
                        if FLAGS.ignore_big and (
                            image_data.shape[0] > 1500 or image_data.shape[1] > 1500
                        ):
                            global SKIPPED  # noqa
                            SKIPPED += 1
                            logging.info(f"Skipping large image; total number skipped: {SKIPPED}")
                            continue
                    except StopIteration:
                        done = True
                        break
                    image_data = jnp.array(image_data)
                    image = Image(
                        data=image_data,
                        shape=image_data.shape,
                        channels=image_data.shape[-1],
                        maxsize="auto",
                    )
                    images.append((image, image_s))
                images = sorted(images, key=lambda x: x[0].shape[0] * x[0].shape[1])

                yield from images
                del images

                if done:
                    break

        generator = sorted_image_gen()
        yield from generator

    elif dataset_name == "ablation":
        # this one loads all the files from a folder, effectively the same as imagenette folder,
        # but does not sort images, assumes they are constant size
        def ablation_image_gen():
            path = Path("Image datasets") / Path(C().dataset)
            assert path.exists(), f"Path {path} does not exist"
            for file in path.rglob("*"):
                if not str(file).lower().endswith((".png", ".jpeg", ".jpg")):
                    continue
                try:
                    image_data = plt.imread(file)  # type: ignore
                    image_shape = image_data.shape[:2]
                    if image_data.ndim == 2:
                        image_data = image_data[..., None]
                    channels = C().out_dim
                    if image_data.max() > 2:
                        image_data = jnp.array(image_data, dtype=jnp.float32) / 255.0
                    image = Image(
                        data=jnp.array(image_data),
                        shape=jnp.array(image_shape),
                        channels=channels,
                        maxsize="auto",
                    )
                    yield image, file
                except Exception as e:
                    logging.error(f"Error loading file {file}: {e!s}")
                    if FLAGS.on_error == "skip":
                        global SKIPPED  # noqa
                        SKIPPED += 1
                        logging.info(f"Skipping file {file}; total skipped: {SKIPPED}")
                        continue
                    elif FLAGS.on_error == "stop":
                        raise
                    elif FLAGS.on_error == "attempt":
                        logging.warning(f"Attempting to fix file {file}, but not implemented.")
                        continue

        generator = ablation_image_gen()
        yield from generator

    else:
        _msg = f"Dataset {dataset_name} not implemented."
        raise NotImplementedError(_msg)


def batchify(generator, batch_size):
    # load images, if image is smaller than 1000x1000 batch it with batch_size other images,
    # otherwise batch it with batch_size / 4 other images

    while True:
        max_h = 0
        max_w = 0
        image_batch = []
        pathes = []
        while len(image_batch) < batch_size:
            try:
                image, path = next(generator)
            except StopIteration:
                break
            path = make_target_path(path)

            if path.exists() and not FLAGS.overwrite:
                continue

            max_w = max(max_w, image.shape[0])
            max_h = max(max_h, image.shape[1])

            image_batch.append(image)
            pathes.append(path)

        logging.info(f"Enlarging to ({max_w}, {max_h})...")
        # now we wish to transform the images into Images with placeholder being the max_size
        for i in range(len(image_batch)):
            image_batch[i] = image_batch[i].enlarge((max_w, max_h))

        stacked_data = jnp.stack([image.data for image in image_batch])
        stacked_shape = jnp.stack([image.shape for image in image_batch])

        assert stacked_shape.shape[0] == stacked_data.shape[0]

        image_soa = Image(data=stacked_data, shape=stacked_shape, channels=3)
        if (
            not (
                jnp.all(image_soa.shape[:, 0] == image_soa.shape[0, 0])
                and jnp.all(image_soa.shape[:, 1] == image_soa.shape[0, 1])
            )
            or batch_size == 1
        ):
            image_soa = image_soa.shrink_to_grid()
        else:
            logging.warning("Using no grid shrinkage: all images must be the same size.")

        efficiency_gap = 0
        max_w, max_h = image_soa.max_shape()
        for i in range(len(image_batch)):
            im_shape = image_soa.shape[i]
            efficiency_gap += max_w * max_h - int(im_shape[0]) * int(im_shape[1])
        efficiency_gap /= len(image_batch)
        efficiency_gap = efficiency_gap / (max_w * max_h)
        logging.info(f"Efficiency: {1.0 - efficiency_gap}")
        yield image_soa, pathes


def bench_dataset(dataset_name):
    total = 0
    batch_size = FLAGS.batch_size
    if dataset_name == "mnist":
        total = 60000
    elif dataset_name == "imagenette":
        total = 14000  # ish
    else:
        total = len(list((Path("Image datasets") / Path(C().dataset)).rglob("*")))

    num_devices = jax.device_count()
    batch_size = batch_size * num_devices
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    sharding = jshard.PositionalSharding(devices)

    datagen_single = data_loader(dataset_name)
    batched_datagen = batchify(datagen_single, batch_size)

    frozen_decoder = None
    if C().dec_shared_at_step != -1:
        logging.info("Starting decoder pretraining...")
        frozen_decoder = train_decoder(
            datagen_single, C().dec_shared_at_step, key=jr.key(42), epochs=C().num_steps
        )
        logging.info("Decoder pretraining done.")

        # reinstantiate the generators, as we've exhausted a bunch of images
        datagen_single = data_loader(dataset_name)
        batched_datagen = batchify(datagen_single, batch_size)

    epochs_list = []
    abs_epochs = [0, *C().save_intermittently_at, C().num_steps]
    for i in range(len(abs_epochs) - 1):
        delta = abs_epochs[i + 1] - abs_epochs[i]
        epochs_list.append(delta)

    logging.info(f"Using effective epoch deltas: {epochs_list}")

    fn = eqx.Partial(train_image, epochs_list=epochs_list, decoder=frozen_decoder)

    idx = 0
    for image_batch, image_path in tqdm(batched_datagen, total=total // batch_size):
        key = jr.key(idx := idx + 1)
        logging.info("Training...")

        dynamic, static = eqx.partition(image_batch, eqx.is_inexact_array)  # only 'data'
        dynamic = jax.lax.with_sharding_constraint(dynamic, sharding)
        image_sharded = eqx.combine(dynamic, static)  # rebuild

        models_list = eqx.filter_vmap(fn)(
            image_sharded, jr.split(key, image_sharded.shape.shape[0])
        )
        logging.info("... done; evaluating...")

        preds = []
        for i, m in enumerate(models_list):
            psnr, _preds = eqx.filter_vmap(eval_image)(m, image_sharded)
            logging.info(f"PSNR@{i}: %.2f +- %.2f", float(psnr.mean()), float(psnr.std()))
            preds.append(_preds)

        with jax.ensure_compile_time_eval():
            for i in range(batch_size):
                # partition the model
                res_dict = {}
                preds_dict = {}
                for ep, model, pred in zip(abs_epochs[1:], [*models_list], [*preds]):
                    model_i = jax.tree.map(lambda x: x[i], model, is_leaf=eqx.is_array)
                    pred_i = pred[i]
                    res_dict[str(ep)] = model_i
                    preds_dict[str(ep)] = pred_i
                record_results(res_dict, preds_dict, image_path[i])


def main(argv):
    del argv  # Unused.
    # Configure absl logging
    logging.get_absl_handler().python_handler.stream = sys.stdout
    if jax.process_index() == 0:
        logging.set_verbosity(logging.INFO)
    else:
        logging.set_verbosity(logging.ERROR)

    if FLAGS.config is not None:
        load_config_from_py(FLAGS.config)

    if FLAGS.task == "trial":
        trial_run()
    elif FLAGS.task == "mnist":
        bench_dataset("mnist")
    elif FLAGS.task == "imagenette":
        bench_dataset("imagenette")
    elif FLAGS.task == "ablation":
        bench_dataset("ablation")
    else:
        _msg = f"Task {FLAGS.task} not implemented. This is... weird. Ask knyazer about it"
        logging.error(_msg)


if __name__ == "__main__":
    app.run(main)
