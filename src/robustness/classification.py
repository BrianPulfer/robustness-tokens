import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from accelerate import Accelerator
import pytorch_lightning as pl

from attacks.pgd import pgd_attack
from data.utils import get_loaders_fn
from models.utils import get_model
from utils import read_config


def build_classifier(
    backbone_path, backbone_kwargs, head_path, head_classifier_name, device="cuda"
):
    backbone = get_model(**backbone_kwargs).to(device)
    if os.path.isfile(backbone_path):
        backbone_sd = torch.load(backbone_path, map_location=device)
        backbone.load_state_dict(backbone_sd)
    backbone.eval()

    head_sd = torch.load(head_path, map_location=device)
    head_w = head_sd["model"][f"classifiers_dict.{head_classifier_name}.linear.weight"]
    head_b = head_sd["model"][f"classifiers_dict.{head_classifier_name}.linear.bias"]

    dim = backbone.model.cls_token.shape[-1]

    head = nn.Linear(2 * dim, 1000, bias=True)
    head.weight.data = head_w
    head.bias.data = head_b
    head = head.to(device)

    return ImageNetClassifier(backbone, head).to(device).eval()


class ImageNetClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        fts = self.backbone(x)
        fts = torch.cat([fts[:, 0], fts[:, 1:].mean(dim=1)], dim=1)
        out = self.head(fts)
        return out


def evaluate_robustness_classification(surrogate, victim, loader, accelerator):
    # Preparing accelerator
    surrogate, victim, loader = accelerator.prepare(surrogate, victim, loader)

    acc, flipped = [], []

    for batch, labels in tqdm(loader, desc="Evaluating robustness"):
        batch_adv = pgd_attack(surrogate, batch, target=labels)

        with torch.no_grad():
            pred = victim(batch)
            pred_adv = victim(batch_adv)
            acc.extend((pred_adv.argmax(dim=-1) == labels).cpu().float().numpy())
            flipped.extend(
                (pred.argmax(dim=-1) != pred_adv.argmax(dim=-1)).cpu().float().numpy()
            )
            print(f"Accuracy: {np.mean(acc):.3f} - Flipped: {np.mean(flipped):.3f}")

    return {"Accuracy": acc}, {"Flipped": flipped}


def main(args):
    # Setting seed
    pl.seed_everything(args["seed"])

    # Accelerator
    accelerator = Accelerator()

    # Data
    loaders_fn = get_loaders_fn(args["dataset"])
    _, val_loader = loaders_fn(args["batch_size"], args["num_workers"])

    # Model
    surrogate = build_classifier(
        args["surrogate"]["backbone_path"],
        args["surrogate"]["backbone_kwargs"],
        args["surrogate"]["head_path"],
        args["surrogate"]["head_classifier_name"],
    )

    victim = build_classifier(
        args["victim"]["backbone_path"],
        args["victim"]["backbone_kwargs"],
        args["victim"]["head_path"],
        args["victim"]["head_classifier_name"],
    )

    # Evaluating robustness in classification
    acc, flipped = evaluate_robustness_classification(
        surrogate, victim, val_loader, accelerator
    )

    # Saving metrics
    rdir = args["results_dir"]
    os.makedirs(rdir, exist_ok=True)
    acc = pd.DataFrame.from_dict(acc)
    acc.to_csv(os.path.join(rdir, "acc.csv"))
    flipped = pd.DataFrame.from_dict(flipped)
    flipped.to_csv(os.path.join(rdir, "flipped.csv"))
    print(f"Robustness metrics saved in {rdir}")


if __name__ == "__main__":
    main(read_config())
