import os
import random
import numpy as np
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

    cossims = []
    mses = []

    def cossim(f1, f2):
        out = cosine_similarity(f1, f2)
        dim = list(range(1, out.ndim))
        return out.mean(dim=dim).cpu().numpy()

    def mse(f1, f2):
        dim = list(range(1, f1.ndim))
        return (f1 - f2).pow(2).mean(dim=dim).cpu().numpy()

    for batch in tqdm(loader, desc="Evaluating robustness"):
        batch = batch[0]
        batch_adv = attack_fn(surrogate, batch)

        with torch.no_grad():
            f1 = victim(batch)
            f2 = victim(batch_adv)
            cossims.extend(cossim(f1, f2))
            mses.extend(mse(f1, f2))
            print(f"Cosine Sim: {np.mean(cossims):.3f}  - MSE: {np.mean(mses):.3f}")

    return {"Cosine Sim": cossims}, {"MSEs": mses}


def main(args):
    # Setting seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])

    # Accelerator
    accelerator = Accelerator()

    # Data
    loaders_fn = get_loaders_fn(args["dataset"])
    _, val_loader = loaders_fn(args["batch_size"], args["num_workers"])

    # Surrogate
    surrogate = get_model(**args["surrogate"])
    if args.get("surrogate_state_dict", None) is not None:
        surrogate.load_state_dict(
            torch.load(args["surrogate_state_dict"], map_location=accelerator.device)
        )

    # Victim
    victim = get_model(**args["victim"])
    if args.get("victim_state_dict", None) is not None:
        victim.load_state_dict(
            torch.load(args["victim_state_dict"], map_location=accelerator.device)
        )

    # Moving to device
    attack_fn = get_attack(**args["attack"])

    # Attacking "robust" model
    cossims, mses = evaluate_robustness(
        surrogate, victim, val_loader, attack_fn, accelerator
    )

    # Saving metrics
    rdir = args["results_dir"]
    os.makedirs(rdir, exist_ok=True)
    cossims = pd.DataFrame.from_dict(cossims)
    mses = pd.DataFrame.from_dict(mses)
    cossims.to_csv(os.path.join(rdir, "cossims.csv"))
    mses.to_csv(os.path.join(rdir, "mses.csv"))
    print(f"Robustness metrics saved in {rdir}")


if __name__ == "__main__":
    main(read_config())
