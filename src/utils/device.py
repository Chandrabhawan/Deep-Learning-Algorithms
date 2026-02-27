"""Device detection and tensor movement helpers."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        torch.device for 'cuda' or 'cpu'.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def to_device(data, device: torch.device):
    """Move tensor(s) to the specified device.

    Handles single tensors, tuples, and lists recursively.
    """
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data
