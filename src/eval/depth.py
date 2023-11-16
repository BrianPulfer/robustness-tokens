import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import read_config
from data.utils import get_loaders_fn
from models.utils import get_model

# from eval.utils.adabins import


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class DepthEstimationModel(pl.LightningModule):
    def __init__(
        self,
        backbone,
        lr=1e-3,
        optim_fn=Adam,
        n_bins=256,
        min_depth=0.001,
        max_depth=10.0,
    ):
        super(DepthEstimationModel, self).__init__()
        self.lr = lr
        self.optim_fn = optim_fn
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.criterion = RMSELoss()

        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.patch_size = backbone.model.patch_size

        dim = backbone.model.norm.weight.shape[0]
        self.head = nn.Sequential(
            nn.Conv2d(2 * dim, n_bins, (1, 1)),
            nn.Softmax(dim=1),
        )
        self.bin_width_estimator = nn.Sequential(
            nn.Linear(dim, n_bins),
            nn.LeakyReLU(),
            nn.Linear(n_bins, n_bins),
            nn.LeakyReLU(),
            nn.Linear(n_bins, n_bins),
            nn.ReLU(),
        )

    def forward(self, x):
        # Sizes
        b, c, h, w = x.shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        # Running through pre-trained backbone
        tokens = self.backbone(x)

        # Concatenating class token to all patches
        sl, d = tokens.shape[1:]
        cls_token, patch_tokens = tokens[:, 0], tokens[:, 1:]
        patch_tokens = torch.stack(
            [torch.cat((cls_token, patch_tokens[:, i]), dim=-1) for i in range(sl - 1)],
            dim=1,
        )

        # Reshaping
        patch_tokens = patch_tokens.reshape(b, h_patches, w_patches, 2 * d).permute(
            0, 3, 1, 2
        )

        # Upsampling each patch by a factor of 4
        patch_tokens = nn.functional.interpolate(
            patch_tokens, scale_factor=4, mode="bilinear", align_corners=False
        )

        # Getting logits for each bin
        logits = self.head(patch_tokens)

        # Getting prediciton of bin widths
        bin_widths = self.bin_width_estimator(cls_token) + 0.1
        bin_widths = bin_widths / bin_widths.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bin_widths
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_depth
        )

        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        out = torch.sum(logits * centers, dim=1, keepdim=True)

        # Upsampling to output size
        out = nn.functional.interpolate(
            out, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze()

        # Clamping depths
        out = out.clamp(min=self.min_depth, max=self.max_depth)

        return out

    def get_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("test_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optim = self.optim_fn(self.head.parameters(), lr=self.lr)
        return optim


def main(args):
    # Setting seed
    pl.seed_everything(args["seed"])

    # Getting data
    image_size = (args["train"]["image_height"], args["train"]["image_width"])
    loaders_fn = get_loaders_fn("nyud")
    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"], args["train"]["num_workers"], image_size=image_size
    )

    # Backbone
    backbone = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=False,
        n_rtokens=args["model"]["n_rtokens"],
    )
    sd = torch.load(args["model"]["checkpoint_path"], map_location="cpu")
    backbone.load_state_dict(sd)

    # Creating depth-estimation model
    optim_fn = getattr(torch.optim, args["train"]["optimizer"])
    model = DepthEstimationModel(backbone, lr=args["train"]["lr"], optim_fn=optim_fn)

    # Training
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
