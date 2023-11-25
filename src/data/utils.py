from data.imagenet import get_loaders as imagenet_loaders


SUPPORTED_DATASETS = ["imagenet"]


def get_loaders_fn(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Supported datasets: {SUPPORTED_DATASETS}"
        )

    if dataset_name == "imagenet":
        return imagenet_loaders

    return None
