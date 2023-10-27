import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.transforms import default_transform


def get_loaders(batch_size, num_workers=0):
    train_set = load_dataset("frgfm/imagenette", "full_size", split="train")
    val_set = load_dataset("frgfm/imagenette", "full_size", split="validation")

    def map_fn(sample):
        return {"image": default_transform(sample["image"].convert("RGB"))}

    train_set = train_set.map(map_fn)
    val_set = val_set.map(map_fn)

    train_set.set_format(type="torch")
    val_set.set_format(type="torch")

    def collate_fn(samples):
        return torch.stack([s["image"] for s in samples]), torch.stack(
            [s["label"] for s in samples]
        )

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

    return train_loader, val_loader
