import tarfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import requests
from absl import logging
from PIL import Image
from sklearn.datasets import fetch_openml
from tqdm import tqdm

# Constants for dataset paths and URLs
DATA_DIR = Path("data")
IMAGENETTE_DIR = Path("data/imagenette2")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGENETTE_TGZ = Path("imagenette2.tgz")


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to the destination path with a progress bar."""
    if dest_path.exists():
        logging.info("File %s already exists. Skipping download.", dest_path)
        return
    logging.info("Downloading %s to %s...", url, dest_path)
    response = requests.get(url, stream=True, timeout=20)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="iB", unit_scale=True, desc=dest_path.name) as pbar:  # noqa
        with dest_path.open("wb") as f:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
    if total_size != 0 and pbar.n != total_size:  # noqa
        raise RuntimeError("ERROR: Download incomplete.")
    logging.info("Download completed: %s", dest_path)


def extract_tar_gz(tar_path: Path, extract_dir: Path) -> None:
    """Extracts a tar.gz file to the specified directory."""
    if extract_dir.exists():
        logging.info("Directory %s already exists. Skipping extraction.", extract_dir)
        return
    logging.info("Extracting %s to %s...", tar_path, extract_dir.parent)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir.parent, filter="data")
    logging.info("Extraction complete.")
    tar_path.unlink()  # Remove the tar.gz file after extraction


def prepare_datasets() -> None:
    """Ensures that MNIST and Imagenette datasets are downloaded and extracted."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare MNIST
    logging.info("Fetching MNIST dataset...")
    _mnist = fetch_openml("mnist_784", version=1, cache=True)
    logging.info("MNIST dataset fetched.")

    # Prepare Imagenette
    if not IMAGENETTE_DIR.exists():
        download_file(IMAGENETTE_URL, IMAGENETTE_TGZ)
        extract_tar_gz(IMAGENETTE_TGZ, IMAGENETTE_DIR)
    else:
        logging.info("Imagenette dataset already exists at %s.", IMAGENETTE_DIR)


def preprocess_mnist(mnist_data: Any) -> tuple[np.ndarray, np.ndarray]:
    X = mnist_data.data.astype(np.float32) / 255.0  # Normalize to [0,1]
    y = mnist_data.target.astype(np.int32)
    return X, y


def load_imagenette_images():
    if not IMAGENETTE_DIR.exists():
        download_file(IMAGENETTE_URL, IMAGENETTE_TGZ)
        extract_tar_gz(IMAGENETTE_TGZ, IMAGENETTE_DIR)
    else:
        logging.info("Imagenette dataset already exists at %s.", IMAGENETTE_DIR)
    for img_path in IMAGENETTE_DIR.rglob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                with Image.open(img_path) as _img:
                    img = _img.convert("RGB")
                    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
                    yield (img_array, img_path)
            except Exception as e:
                logging.warning("Failed to load image %s: %s", img_path, e)
