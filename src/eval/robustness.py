import os

import pandas as pd
import torch
from accelerate import Accelerator
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm

from attacks import pgd_attack
from data.utils import get_loaders_fn
from models.utils import get_model
from utils import read_config


def evaluate_rtokens(model, loader, attack_fn, accelerator):
    """Test loop that evaluates robustness of the model compared to the baseline."""
    # Preparing accelerator
    model, loader = accelerator.prepare(model, loader)

    modes = ["Standard", "Robust"]
    cossims = {m1 + "-" + m2: [] for m1 in modes for m2 in modes}
    mses = {m1 + "-" + m2: [] for m1 in modes for m2 in modes}

    def cossim(f1, f2):
        out = cosine_similarity(f1, f2)
        dim = list(range(1, out.ndim))
        return out.mean(dim=dim).cpu().numpy()

    def mse(f1, f2):
        dim = list(range(1, f1.ndim))
        return (f1 - f2).pow(2).mean(dim=dim).cpu().numpy()

    for batch in tqdm(loader, desc="Evaluating robustness"):
        batch = batch[0]
        for b1, mode1 in zip([True, False], modes):
            model.enable_robust = b1
            batch_adv = attack_fn(model, batch)

            with torch.no_grad():
                for b2, mode2 in zip([True, False], modes):
                    model.enable_robust = b2
                    f1 = model(batch)
                    f2 = model(batch_adv)
                    cossims[mode1 + "-" + mode2].extend(cossim(f1, f2))
                    mses[mode1 + "-" + mode2].extend(mse(f1, f2))

    return cossims, mses


def main(args):
    # Accelerator
    accelerator = Accelerator()

    # Data
    loaders_fn = get_loaders_fn(args["dataset"])
    _, val_loader = loaders_fn(args["batch_size"], args["num_workers"])

    # Model
    model = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=args["model"]["return_cls"],
        n_rtokens=args["model"]["n_rtokens"],
    )

    model, val_loader = accelerator.prepare(model, val_loader)

    model.load_state_dict(
        torch.load(args["state_dict"], map_location=accelerator.device)
    )

    criterion = getattr(torch.nn, args["train"]["criterion"])()

    def attack_fn(model, batch):
        mode = model.enable_robust
        model.enable_robust = False
        batch_adv = pgd_attack(
            model,
            batch,
            steps=args["attack"]["steps"],
            lr=args["attack"]["lr"],
            eps=args["attack"]["eps"],
            max_mse=args["attack"]["max_mse"],
        )
        model.enable_robust = mode
        return batch_adv

    # Attacking "robust" model
    accelerator = Accelerator()
    cossims, mses = evaluate_rtokens(
        model, val_loader, criterion, attack_fn, accelerator
    )

    # Saving metrics
    rdir = args["results_dir"]
    cossims = pd.DataFrame.from_dict(cossims)
    mses = pd.DataFrame.from_dict(mses)
    cossims.to_csv(os.path.join(rdir, "cossims.csv"))
    mses.to_csv(os.path.join(rdir, "mses.csv"))
    print(f"Robustness metrics saved in {rdir}")


if __name__ == "__main__":
    main(read_config())
