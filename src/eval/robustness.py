import os

import pandas as pd
import torch
from accelerate import Accelerator
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm

from attacks.utils import get_attack
from data.utils import get_loaders_fn
from models.utils import get_model
from utils import read_config


def evaluate_robustness(surrogate, victim, loader, attack_fn, accelerator):
    """Test loop that evaluates robustness of the model compared to the baseline."""
    # Preparing accelerator
    surrogate, victim, loader = accelerator.prepare(surrogate, victim, loader)

    modes = ["Robust", "Standard"]
    cossims = {"Cossims " + m1 + "-" + m2: [] for m1 in modes for m2 in modes}
    mses = {"MSEs " + m1 + "-" + m2: [] for m1 in modes for m2 in modes}

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
            surrogate.enable_robust = b1
            batch_adv = attack_fn(surrogate, batch)

            with torch.no_grad():
                for b2, mode2 in zip([True, False], modes):
                    victim.enable_robust = b2
                    f1 = victim(batch)
                    f2 = victim(batch_adv)
                    cossims["Cossims " + mode1 + "-" + mode2].extend(cossim(f1, f2))
                    mses["MSEs " + mode1 + "-" + mode2].extend(mse(f1, f2))

    return cossims, mses


def main(args):
    # Accelerator
    accelerator = Accelerator()

    # Data
    loaders_fn = get_loaders_fn(args["dataset"])
    _, val_loader = loaders_fn(args["batch_size"], args["num_workers"])

    # Surrogate
    surrogate = get_model(**args["surrogate"])
    if args["surrogate"].get("state_dict", None) is not None:
        surrogate.load_state_dict(
            torch.load(args["surrogate"]["state_dict"], map_location=accelerator.device)
        )

    # Victim
    victim = get_model(**args["victim"])
    if args["victim"].get("state_dict", None) is not None:
        victim.load_state_dict(
            torch.load(args["victim"]["state_dict"], map_location=accelerator.device)
        )

    # Moving to device
    attack_fn = get_attack(**args["attack"])

    # Attacking "robust" model
    cossims, mses = evaluate_robustness(
        surrogate, victim, val_loader, attack_fn, accelerator
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
