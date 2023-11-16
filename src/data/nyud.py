from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize
from datasets import load_dataset

from data.transforms import to_tensor, normalize


class NYUdSet(Dataset):
    def __init__(self, split="train", image_size=(480, 640)):
        super(Dataset, self).__init__()
        self.split = split
        self.dataset = load_dataset("sayakpaul/nyu_depth_v2", split=self.split)

        self.img_transform = Compose([Resize(image_size), to_tensor, normalize])
        self.mask_transform = Compose([Resize(image_size), to_tensor])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        mask = self.dataset[idx]["depth_map"]

        img = self.img_transform(img)
        mask = self.mask_transform(mask).squeeze()

        return img, mask


def get_loaders(batch_size, num_workers=4, image_size=(448, 448)):
    train_set = NYUdSet("train", image_size)
    val_set = NYUdSet("validation", image_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
