"""Reproducibility: set global random seeds for torch, numpy, and random."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
