import tarfile
from pathlib import Path

import requests
from absl import logging
from sklearn.datasets import fetch_openml
from tqdm import tqdm

# Constants for dataset paths and URLs
DATA_DIR = Path("data")
IMAGENETTE_DIR = Path("imagenette2")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_TGZ = Path("imagenette2-320.tgz")


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to the destination path with a progress bar."""
    if dest_path.exists():
        logging.info("File %s already exists. Skipping download.", dest_path)
        return
    logging.info("Downloading %s to %s...", url, dest_path)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="iB", unit_scale=True, desc=dest_path.name) as pbar:
        with open(dest_path, "wb") as f:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
    if total_size != 0 and pbar.n != total_size:
        raise Exception("ERROR: Download incomplete.")
    logging.info("Download completed: %s", dest_path)


def extract_tar_gz(tar_path: Path, extract_dir: Path) -> None:
    """Extracts a tar.gz file to the specified directory."""
    if extract_dir.exists():
        logging.info("Directory %s already exists. Skipping extraction.", extract_dir)
        return
    logging.info("Extracting %s to %s...", tar_path, extract_dir.parent)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir.parent)
    logging.info("Extraction complete.")
    tar_path.unlink()  # Remove the tar.gz file after extraction


def prepare_datasets() -> None:
    """Ensures that MNIST and Imagenette datasets are downloaded and extracted."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare MNIST
    logging.info("Fetching MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, cache=True)
    logging.info("MNIST dataset fetched.")

    # Prepare Imagenette
    if not IMAGENETTE_DIR.exists():
        download_file(IMAGENETTE_URL, IMAGENETTE_TGZ)
        extract_tar_gz(IMAGENETTE_TGZ, IMAGENETTE_DIR)
    else:
        logging.info("Imagenette dataset already exists at %s.", IMAGENETTE_DIR)
