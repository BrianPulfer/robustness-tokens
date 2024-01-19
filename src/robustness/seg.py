import os
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from utils import read_config
from attacks.pgd import pgd_attack

from mmcv.utils import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor as build_segmentor_mmseg
from mmseg.datasets import build_dataset, build_dataloader

DATA_DICT = dict(
    type="ADE20KDataset",
    data_root="/srv/beegfs/scratch/users/p/pulfer/datasets/ADE20K/ADEChallengeData2016",
    img_dir="images/training",
    ann_dir="annotations/training",
    pipeline=[
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", reduce_zero_label=True),
        dict(type="Resize", img_scale=(99999999, 518), ratio_range=(0.5, 2.0)),
        dict(type="RandomCrop", crop_size=(518, 518), cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(
            type="Normalize",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        ),
        dict(type="Pad", size=(518, 518), pad_val=0, seg_pad_val=255),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    ],
)


def build_segmentor(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor_mmseg(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    load_checkpoint(model, ckpt_path, map_location="cpu")
    return model


def evaluate_robustness_segmentation(surrogate, victim, loader, device):
    surrogate = surrogate.to(device)
    victim = victim.to(device)

    accs, ces = [], []
    accs_adv, ces_adv = [], []

    for batch in loader:
        # Unpack batch
        img = batch["img"].data[0].to(device)
        img_metas = batch["img_metas"]
        gt_semantic_seg = batch["gt_semantic_seg"].data[0].to(device)

        # Getting adversarial perturbation
        img_adv = pgd_attack(surrogate, img, gt_semantic_seg)

        # Forward pass
        with torch.no_grad():
            pred = victim(img, img_metas=img_metas, gt_semantic_seg=gt_semantic_seg)
            accs.append(pred["decode.acc_seg"].item())
            ces.append(pred["decode.loss_ce"].item())

            pred_adv = victim(
                img_adv, img_metas=img_metas, gt_semantic_seg=gt_semantic_seg
            )
            accs_adv.append(pred_adv["decode.acc_seg"].item())
            ces_adv.append(pred_adv["decode.loss_ce"].item())

    return accs, ces, accs_adv, ces_adv


def main(args):
    # Setting random seed
    pl.seed_everything(0)

    # Loading dataset
    dataset = build_dataset(DATA_DICT)
    loader = build_dataloader(
        dataset, samples_per_gpu=args["samples_per_gpu"], workers_per_gpu=4, dist=False
    )

    # Building MMSegmentation Models
    surrogate = build_segmentor(
        args["surrogate"]["cfg_path"], args["surrogate"]["ckpt_path"]
    )
    victim = build_segmentor(args["victim"]["cfg_path"], args["victim"]["ckpt_path"])

    # Measure model performance in terms of mIoU (should be >0.4)
    device = torch.device("cuda")
    accs, ces, accs_adv, ces_adv = evaluate_robustness_segmentation(
        surrogate, victim, loader, device
    )

    # Storing results
    pd.DataFrame.from_dict(
        {
            "Accuracy original": np.mean(accs),
            "CE Loss original": np.mean(ces),
            "Accuracy adversarial": np.mean(accs_adv),
            "CE Loss adversarial": np.mean(ces_adv),
        }
    ).to_csv(os.path.join(args["result_dir"], "robustness_segmentation.csv"))


if __name__ == "__main__":
    main(read_config())
