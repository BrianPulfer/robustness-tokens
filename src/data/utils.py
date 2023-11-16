from data.ade20k import get_loaders as ade20k_loaders
from data.imagenet import get_loaders as imagenet_loaders
from data.imagenette import get_loaders as imagenette_loaders
from data.nyud import get_loaders as nyud_loaders


SUPPORTED_DATASETS = ["ade20k", "imagenet", "imagenette", "nyud"]


def get_loaders_fn(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS}"
        )

    if dataset_name == "ade20k":
        return ade20k_loaders

    if dataset_name == "imagenet":
        return imagenet_loaders

    if dataset_name == "imagenette":
        return imagenette_loaders

    if dataset_name == "nyud":
        return nyud_loaders

    return None
