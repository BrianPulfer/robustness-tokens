import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils import read_config
from attacks.pgd import pgd_attack

from mmcv.utils import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor as build_segmentor_mmseg
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models.losses import DiceLoss

DATA_DICT = dict(
    type="ADE20KDataset",
    data_root="/srv/beegfs/scratch/users/p/pulfer/datasets/ADE20K/ADEChallengeData2016",
    img_dir="images/validation",
    ann_dir="annotations/validation",
    pipeline=[
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", reduce_zero_label=True),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(99999999, 518),
            img_ratios=1.0,
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomCrop", crop_size=(518, 518), cat_max_ratio=1.0),
                dict(type="RandomFlip"),
                dict(
                    type="Normalize",
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True,
                ),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img", "gt_semantic_seg"]),
            ],
        ),
    ],
)


class SegmentorWrapper(nn.Module):
    def __init__(self, cfg_path, ckpt_path):
        super(SegmentorWrapper, self).__init__()
        self.cfg = Config.fromfile(cfg_path)
        self.model = build_segmentor_mmseg(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        load_checkpoint(self.model, ckpt_path, map_location="cpu")

    def forward(self, img, img_metas):
        return self.model.encode_decode(img, img_metas=img_metas)


def evaluate_robustness_segmentation(surrogate, victim, loader, device):
    surrogate = surrogate.to(device)
    victim = victim.to(device)

    ign_idx = 255
    miou = DiceLoss(ignore_index=ign_idx).to(device)
    mious, mious_adv = [], []

    for batch in tqdm(loader, desc="Evaluating robustness"):
        # Unpack batch
        img = batch["img"][0].to(device)
        img_metas = batch["img_metas"]
        gt_semantic_seg = batch["gt_semantic_seg"][0].long().to(device)
        forward_kwargs = dict(img_metas=img_metas)

        # Getting adversarial perturbation
        img_adv = pgd_attack(
            surrogate, img, gt_semantic_seg, ignore_index=ign_idx, **forward_kwargs
        )

        # Forward pass
        with torch.no_grad():
            pred = victim(img, **forward_kwargs)
            mious.extend(
                [
                    miou(p.unsqueeze(0), gt.unsqueeze(0)).item()
                    for p, gt in zip(pred, gt_semantic_seg)
                ]
            )

            pred_adv = victim(img_adv, **forward_kwargs)
            mious_adv.extend(
                [
                    miou(p.unsqueeze(0), gt.unsqueeze(0)).item()
                    for p, gt in zip(pred_adv, gt_semantic_seg)
                ]
            )
        print(f"mIoU: {np.mean(mious):.3f}  - mIoU adv: {np.mean(mious_adv):.3f}")

    return mious, mious_adv


def main(args):
    # Setting random seed
    pl.seed_everything(0)

    # Loading dataset
    dataset = build_dataset(DATA_DICT)
    loader = build_dataloader(
        dataset, samples_per_gpu=args["samples_per_gpu"], workers_per_gpu=4, dist=False
    )

    # Building MMSegmentation Models
    surrogate = SegmentorWrapper(
        args["surrogate"]["cfg_path"], args["surrogate"]["ckpt_path"]
    )
    victim = SegmentorWrapper(args["victim"]["cfg_path"], args["victim"]["ckpt_path"])

    # Measure model performance in terms of mIoU (should be >0.4)
    device = torch.device("cuda")
    mious, mious_adv = evaluate_robustness_segmentation(
        surrogate, victim, loader, device
    )

    # Storing results
    rdir = args["results_dir"]
    os.makedirs(rdir, exist_ok=True)
    pd.DataFrame.from_dict(
        {
            "mIoU Original": np.array(mious),
            "mIoU Adversary": np.array(mious_adv),
        }
    ).to_csv(os.path.join(rdir, "mIoUs.csv"))
    print(f"Robustness metrics saved in {rdir}")


if __name__ == "__main__":
    args = read_config()
    main(args)
