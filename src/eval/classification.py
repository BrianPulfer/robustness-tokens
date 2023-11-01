import torch
import torch.nn as nn
from torch.optim import Adam

from torchmetrics import Accuracy

from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import read_config
from models.utils import get_model
from data.utils import get_loaders_fn
from data.transforms import to_tensor, normalize


class Classifier(pl.LightningModule):
    def __init__(self, backbone, lr=1e-3, optim_fn=Adam):
        super().__init__()
        self.lr = lr
        self.optim_fn = optim_fn
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", num_classes=1000)

        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)

        dim = backbone.model.norm.weight.shape[0]
        self.head = nn.Linear(2 * dim, 1000)

    def forward(self, x):
        h = self.backbone(x)
        cls_tok, avg_pool = h[:, 0], h[:, 1:].mean(dim=1)
        return self.head(torch.cat((cls_tok, avg_pool), dim=-1))

    def get_loss_and_acc(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.acc(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.get_loss_and_acc(batch)
        self.log_dict(
            {"train_loss": loss.item(), "train_acc": acc.item()}, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.get_loss_and_acc(batch)
        self.log_dict(
            {"test_loss": loss.item(), "test_acc": acc.item()}, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optim = self.optim_fn(self.head.parameters(), lr=self.lr)
        # optim = self.optim_fn(self.trainer.model.head.parameters(), lr=self.lr) # DistributedDataParallel doesn't work with this
        return optim


def main(args):
    """Runs linear evaluation on ImageNet."""

    # Setting seed
    pl.seed_everything(args["seed"])

    # Backbone
    backbone = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=False,
        n_rtokens=args["model"]["n_rtokens"],
    )

    sd = torch.load(args["model"]["checkpoint_path"], map_location="cpu")
    backbone.load_state_dict(sd)

    optim_fn = getattr(torch.optim, args["train"]["optimizer"])
    model = Classifier(backbone, args["train"]["lr"], optim_fn)

    # Dataset
    loaders_fn = get_loaders_fn("imagenet")

    train_transform = Compose(
        [
            RandomResizedCrop(224, scale=(0.5, 1.0)),
            RandomHorizontalFlip(p=0.5),
            to_tensor,
            normalize,
        ]
    )

    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"],
        num_workers=args["train"]["num_workers"],
        train_transform=train_transform,
    )

    # Training
    logger = WandbLogger(project="Robustness Tokens", name=args["run_name"])
    logger.experiment.config.update(args)
    trainer = pl.Trainer(
        accelerator="auto",
        # strategy="ddp",
        strategy="ddp_find_unused_parameters_true",
        max_steps=args["train"]["max_steps"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=args["results_dir"],
                filename="last",
                save_last=True,
                monitor="train_loss",
                every_n_train_steps=args["train"]["checkpoint_freq"],
            )
        ],
    )
    trainer.fit(model, train_loader)

    # Testing (latest model is used)
    trainer.test(model, val_loader)


if __name__ == "__main__":
    main(read_config())
