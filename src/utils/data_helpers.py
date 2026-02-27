"""Data loading helpers with fallback for dataset downloads."""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Return the project data/raw directory, creating it if needed."""
    # Works whether called from notebooks/ or project root
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "raw",
        Path("data") / "raw",
        Path("../data") / "raw",
        Path("../../data") / "raw",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Default: create relative to this file
    default = candidates[0]
    default.mkdir(parents=True, exist_ok=True)
    return default


def safe_download_dataset(dataset_class, root=None, **kwargs):
    """Attempt to download a torchvision/torchaudio dataset with fallback.

    Args:
        dataset_class: A torchvision.datasets class (e.g., MNIST, CIFAR10).
        root: Directory to store data. Defaults to get_data_dir().
        **kwargs: Passed to the dataset constructor.

    Returns:
        Dataset instance, or None if download fails.
    """
    if root is None:
        root = str(get_data_dir())
    try:
        return dataset_class(root=root, download=True, **kwargs)
    except Exception as e:
        print(f"[WARNING] Could not download {dataset_class.__name__}: {e}")
        print("Falling back to sklearn digits or synthetic data.")
        return None
