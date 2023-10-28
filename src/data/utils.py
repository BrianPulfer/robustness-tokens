from data.imagenet import get_loaders as imagenet_loaders
from data.imagenette import get_loaders as imagenette_loaders

SUPPORTED_DATASETS = [
    "imagenet",
    "imagenette",
]


def get_loaders_fn(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS}"
        )

    if dataset_name == "imagenet":
        return imagenet_loaders

    if dataset_name == "imagenette":
        return imagenette_loaders

    return None
