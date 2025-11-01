import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from faiss import knn, METRIC_INNER_PRODUCT
from faiss.contrib.vecs_io import fvecs_read, ivecs_read, ivecs_write


def show_progress(block_num, block_size, total_size):
    print(
        f"Downloading: {round(min(block_num * block_size / total_size, 1) * 100, 2)}%",
        end="\r",
    )


def download(url, zip_file):
    try:
        urlretrieve(url, zip_file, reporthook=show_progress)
        print()
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def extract(zip_file, extract_path):
    try:
        # Extract tar.gz file
        file = tarfile.open(zip_file)
        file.extractall(extract_path, numeric_owner=True)
        file.close()

    except Exception as e:
        raise RuntimeError(f"Failed to extract {zip_file}: {e}") from e


def download_and_extract(url, zip_file, extract_path):
    download(url, zip_file)
    extract(zip_file, extract_path)


def dataset_path(dataset):
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    dataset_dir = PROJECT_ROOT / f"dataset/{dataset}"
    return dataset_dir


def download_dataset(dataset):
    dataset_dir = dataset_path(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_file = dataset_dir / f"{dataset}.tar.gz"

    download_and_extract(
        f"https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/{dataset}.tar.gz",
        zip_file,
        dataset_dir.parent,
    )


def generate_groundtruth(dataset, k=100):
    dataset_dir = dataset_path(dataset)

    _, I = knn(
        load_data(dataset, "query"),
        load_data(dataset, "base"),
        k=k,
        metric=METRIC_INNER_PRODUCT,
    )

    # Ensure write permission for the dataset directory
    os.chmod(dataset_dir, os.stat(dataset_dir).st_mode | 0o200)

    gt_file = dataset_dir / f"{dataset}_gt_k_{k}.ivecs"
    ivecs_write(gt_file, I)


def load_data(dataset, ftype, k=100):
    name = [
        "cifar60k",
        "gist",
        "tiny5m",
        "sift10m",
        "deep1M",
        "msong",
        "glove1.2m",
        "glove2.2m",
        "audio",
        "nuswide",
        "ukbench",
        "imagenet",
        "movielens",
        "netflix",
        "yahoomusic",
        "word2vec",
    ]

    if dataset not in name:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_dir = dataset_path(dataset)
    if ftype == "base":
        file = dataset_dir / f"{dataset}_base.fvecs"
    elif ftype == "query":
        file = dataset_dir / f"{dataset}_query.fvecs"
    elif ftype == "gt":
        file = dataset_dir / f"{dataset}_gt_k_{k}.ivecs"
    else:
        raise ValueError(f"Unknown file type: {ftype}")

    if not file.exists():
        if ftype in ["base", "query"]:
            download_dataset(dataset)
            generate_groundtruth(dataset, k=k)  # k=100 by default

        elif ftype == "gt":
            if (
                not (dataset_dir / f"{dataset}_base.fvecs").exists()
                or not (dataset_dir / f"{dataset}_query.fvecs").exists()
            ):
                download_dataset(dataset)
            generate_groundtruth(dataset, k=k)

    if file.suffix == ".fvecs":
        return fvecs_read(file)
    elif file.suffix == ".ivecs":
        return ivecs_read(file)


def load_base(dataset):
    return load_data(dataset, "base")


def load_query(dataset):
    return load_data(dataset, "query")


def load_groundtruth(dataset, k=100):
    if k < 100:
        k = 100
    return load_data(dataset, "gt", k=k)
