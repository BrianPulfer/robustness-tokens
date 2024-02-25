import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from accelerate import Accelerator
import pytorch_lightning as pl

import torchattacks

from attacks.pgd import pgd_attack
from data.utils import get_loaders_fn
from models.utils import get_model
from utils import read_config

# Whether to test on torchattacks
USE_TORCHATTACKS = True
TORCHATTACKS = ["PGD", "AutoAttack"]


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
    
class BatchMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2, dim=1)


def evaluate_robustness_classification(
    surrogate, victim, loader, accelerator, result_dir
):
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

    # Saving metrics
    acc = pd.DataFrame.from_dict(acc)
    acc.to_csv(os.path.join(result_dir, "acc.csv"))
    flipped = pd.DataFrame.from_dict(flipped)
    flipped.to_csv(os.path.join(result_dir, "flipped.csv"))
    print(f"Robustness metrics saved in {result_dir}")


def evaluate_robustness_classification_torchattacks(
    surrogate, victim, loader, accelerator, result_dir
):
    # Preparing accelerator
    surrogate, victim, loader = accelerator.prepare(surrogate, victim, loader)
    
    # Metrics
    metrics = [("Cossim", nn.CosineSimilarity(dim=1)), ("MSE", BatchMSELoss())]
    columns = [f"{attack}_{metric}_{model}" for attack in TORCHATTACKS for metric, _ in metrics for model in ["surrogate", "victim"]]

    writing = False
    for batch, labels in tqdm(loader, desc="Evaluating robustness"):
        bs = batch.shape[0]
        data = []
        for attack in TORCHATTACKS:
            atk = getattr(torchattacks, attack)(surrogate)
            batch_adv = atk(batch, labels)
            
            with torch.no_grad():
                pred_s = surrogate(batch).view(bs, -1)
                pred_s_adv = surrogate(batch_adv).view(bs, -1)
                pred_v = victim(batch).view(bs, -1)
                pred_v_adv = victim(batch_adv).view(bs, -1)

            for _, metric in metrics:
                value_s = metric(pred_s, pred_s_adv).cpu().numpy()
                value_v = metric(pred_v, pred_v_adv).cpu().numpy()
                data.extend([value_s, value_v])

        pd.DataFrame(
            data=zip(*data),
            columns=columns,
        ).to_csv(
            os.path.join(result_dir, "torchattacks.csv"),
            index=False,
            columns=columns,
            mode="a" if writing else "w",
            header=not writing,
        )
        writing = True


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
    rdir = args["results_dir"]
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "config.yaml"), "w") as f:
        yaml.dump(args, f)

    if USE_TORCHATTACKS:
        evaluate_robustness_classification_torchattacks(
            surrogate, victim, val_loader, accelerator, rdir
        )
    else:
        evaluate_robustness_classification(
            surrogate,
            victim,
            val_loader,
            accelerator,
            rdir,
        )


if __name__ == "__main__":
    main(read_config())
