import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.transforms import transform


def get_loaders(batch_size, num_workers=0):
    train_set = load_dataset("frgfm/imagenette", "full_size", split="train[:70%]")
    test_set = load_dataset("frgfm/imagenette", "full_size", split="train[70%:]")
    val_set = load_dataset("frgfm/imagenette", "full_size", split="validation")

    def map_fn(sample):
        return {"image": transform(sample["image"].convert("RGB"))}

    train_set = train_set.map(map_fn)
    val_set = val_set.map(map_fn)
    test_set = test_set.map(map_fn)

    train_set.set_format(type="torch", columns=["image"])
    val_set.set_format(type="torch", columns=["image"])
    test_set.set_format(type="torch", columns=["image"])

    def collate_fn(samples):
        return torch.stack([s["image"] for s in samples])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
