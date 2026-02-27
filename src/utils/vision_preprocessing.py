"""Vision preprocessing: standard transforms and augmentation pipelines."""

from torchvision import transforms


def get_basic_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    grayscale: bool = False,
) -> transforms.Compose:
    """Return a standard eval/test transform pipeline.

    Args:
        image_size: Resize target (square).
        mean: Channel means for normalization.
        std: Channel stds for normalization.
        grayscale: If True, use single-channel normalization.
    """
    if grayscale:
        mean, std = (0.5,), (0.5,)
    tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(tfms)


def get_augmentation_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    grayscale: bool = False,
) -> transforms.Compose:
    """Return a training transform pipeline with common augmentations.

    Includes random crop, horizontal flip, color jitter, and normalization.
    """
    if grayscale:
        mean, std = (0.5,), (0.5,)
        tfms = [
            transforms.Resize((image_size + 4, image_size + 4)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        tfms = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(tfms)
