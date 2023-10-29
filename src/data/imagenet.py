import os

import dotenv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from data.transforms import default_transform


def get_loaders(
    batch_size,
    num_workers=4,
    train_transform=default_transform,
    val_transform=default_transform,
):
    dotenv.load_dotenv()
    IMAGENET_DIR = os.getenv("IMAGENET_DIR")

    assert IMAGENET_DIR is not None, "IMAGENET_DIR is not set in .env"

    train_set = ImageNet(root=IMAGENET_DIR, split="train", transform=train_transform)
    val_set = ImageNet(root=IMAGENET_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
