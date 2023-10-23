from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


class ImageNetDataset(LightningDataModule):
    def __init__(self, batch_size=128, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
