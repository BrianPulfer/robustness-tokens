import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import make_grid

from utils import read_config
from attacks.pgd import pgd_attack
from data.transforms import to_pil, unnormalize

from mmcv.utils import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor as build_segmentor_mmseg
from mmseg.datasets import build_dataset, build_dataloader


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


def visualize_robustness_segmentation(
    surrogate, victim, loader, device, rdir="visualizations/seg"
):
    os.makedirs(rdir, exist_ok=True)
    surrogate = surrogate.eval().to(device)
    victim = victim.eval().to(device)

    ign_idx = 255
    count = 0
    for batch in tqdm(loader, desc="Storing visualizations"):
        count += 1

        # Unpack batch
        img = batch["img"][0].to(device)
        img_metas = batch["img_metas"]
        gt_semantic_seg = batch["gt_semantic_seg"][0].squeeze().long().to(device)
        forward_kwargs = dict(img_metas=img_metas)

        # Getting adversarial perturbation
        img_adv = pgd_attack(
            surrogate, img, gt_semantic_seg, ignore_index=ign_idx, **forward_kwargs
        )

        # Forward pass
        with torch.no_grad():
            pred = victim(img, **forward_kwargs)
            pred_adv = victim(img_adv, **forward_kwargs)

        for ori_img, adv_img, gt, ori_p, adv_p in zip(
            img, img_adv, gt_semantic_seg, pred, pred_adv
        ):
            # Getting images
            ori_img = unnormalize(ori_img).cpu()
            adv_img = unnormalize(adv_img).cpu()
            gt = gt.repeat(3, 1, 1).cpu()
            ori_p = ori_p.argmax(0).repeat(3, 1, 1).cpu()
            adv_p = adv_p.argmax(0).repeat(3, 1, 1).cpu()

            # Showing them together
            display_img = make_grid([ori_img, ori_p, gt, adv_img, adv_p, gt], nrow=3)
            to_pil(display_img).save(os.path.join(rdir, f"{count}.png"))


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
    visualize_robustness_segmentation(surrogate, victim, loader, device)


if __name__ == "__main__":
    args = read_config()
    main(args)
