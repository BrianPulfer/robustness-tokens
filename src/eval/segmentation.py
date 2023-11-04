import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import read_config
from models.utils import get_model
from data.utils import get_loaders_fn


class IoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        return (y_hat == y).float().mean()


class SegmentationModel(pl.LightningModule):
    def __init__(self, backbone, num_classes=3688, lr=1e-3, optim_fn=Adam):
        super().__init__()

        self.num_classes = num_classes
        self.patch_size = backbone.model.patch_size

        self.lr = lr
        self.optim_fn = optim_fn
        self.criterion = nn.CrossEntropyLoss()
        self.iou = IoU()

        # Backbone network
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)

        # Segmentation head
        dim = backbone.model.norm.weight.shape[0]
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Sizes
        b, c, h, w = x.shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        # Running through pre-trained backbone
        out = self.backbone(x)[:, 1:]

        # Linear classification head per patch
        out = self.head(out)

        # Reshaping and upsampling output
        b, _, d = out.shape
        out = out.reshape(b, h_patches, w_patches, d).permute(0, 3, 1, 2)
        out = torch.nn.functional.interpolate(
            out, size=(h, w), mode="bilinear", align_corners=False
        )
        # out = torch.cat([torch.nn.functional.interpolate(out[i].unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False) for i in range(b)], dim=0)

        return out

    def get_loss_and_iou(self, batch):
        image, label = batch["image"], batch["label"]
        y_hat = self(image)
        loss = self.criterion(y_hat, label)
        iou = self.iou(y_hat, label)
        return loss, iou

    def training_step(self, batch, batch_idx):
        loss, iou = self.get_loss_and_iou(batch)
        self.log_dict(
            {"train_loss": loss.item(), "train_iou": iou.item()}, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, iou = self.get_loss_and_iou(batch)
        self.log_dict(
            {"test_loss": loss.item(), "test_iou": iou.item()}, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optim = self.optim_fn(self.head.parameters(), lr=self.lr)
        return optim


def main(args):
    # Setting random seed
    pl.seed_everything(args["seed"])

    # Loading data
    image_size = (args["train"]["image_size"], args["train"]["image_size"])
    loaders_fn = get_loaders_fn("ade20k")
    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"],
        num_workers=args["train"]["num_workers"],
        image_size=image_size,
    )

    # Loading backbone
    backbone = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=False,
        n_rtokens=args["model"]["n_rtokens"],
    )
    sd = torch.load(args["model"]["checkpoint_path"], map_location="cpu")
    backbone.load_state_dict(sd)

    # Creating segmentation model
    optim_fn = getattr(torch.optim, args["train"]["optimizer"])
    model = SegmentationModel(backbone, lr=args["train"]["lr"], optim_fn=optim_fn)

    # Training segmentation head
    logger = WandbLogger(project="Robustness Tokens", name=args["run_name"])
    logger.experiment.config.update(args)
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp",
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
