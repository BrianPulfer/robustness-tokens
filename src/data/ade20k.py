import os
import dotenv

import torch

from PIL import Image
from mmseg.datasets.ade import ADE20KDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

from data.transforms import to_tensor, normalize


class ADE20K(Dataset):
    def __init__(self, data_root, split="train", image_size=(448, 448)):
        self.dataset = ADE20KDataset(data_root=data_root)
        self.split = "training" if "train" in split.lower() else "validation"
        self.resize = Resize(image_size)
        self.indices = [
            i
            for i in range(len(self.dataset))
            if self.split in self.dataset[i]["img_path"]
        ]

    def __getitem__(self, idx):
        element = self.dataset[self.indices[idx]]

        image_path = element["img_path"]
        label_path = image_path.replace(".jpg", "_seg.png")

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        # Applying default transform
        image = normalize(to_tensor(self.resize(image)))
        label = to_tensor(self.resize(label)) * 255

        # Transforming label
        R, G = label[0, :, :], label[1, :, :]
        label = (R / 10) * 256 + G
        label = label.long()

        return {"image": image, "label": label}

    def __len__(self):
        return len(self.indices)


def get_loaders(batch_size, num_workers=4, image_size=(448, 448)):
    dotenv.load_dotenv()
    ADE20K_DIR = os.getenv("ADE20K_DIR")
    assert ADE20K_DIR is not None, "ADE20K_DIR is not set in .env"

    train_set = ADE20K(ADE20K_DIR, "train", image_size)
    val_set = ADE20K(ADE20K_DIR, "val", image_size)

    def collate_dicts(dicts):
        return {k: torch.stack([d[k] for d in dicts]) for k in dicts[0]}

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dicts,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dicts,
    )

    return train_loader, val_loader
