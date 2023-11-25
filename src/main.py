import os
import yaml
import random

import torch
import wandb
import numpy as np
import pandas as pd
from accelerate import Accelerator

from utils import read_config
from convert import convert_checkpoint
from attacks.utils import get_attack
from models.utils import get_model
from data.utils import get_loaders_fn
from train.robustness import train_rtokens
from eval.robustness import evaluate_robustness


def main(args):
    """Trains robustness tokens for DinoV2 and evaluates robustness."""
    # Setting seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])

    # Initializing wandb
    wandb.init(project="Robustness Tokens", name=args["run_name"], config=args)

    # Creating result directory and copying config file
    os.makedirs(args["results_dir"], exist_ok=True)
    yaml.dump(args, open(os.path.join(args["results_dir"], "config.yaml"), "w"))

    # Initializing model
    model = get_model(**args["model"])

    # Defining attack function
    attack_fn = get_attack(model, **args["attack"])

    # Preparing data loaders
    loaders_fn = get_loaders_fn(args["dataset"])
    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"], num_workers=args["train"]["num_workers"]
    )

    # Training hyper-parameters
    max_steps = args["train"]["max_steps"]
    checkpoint_freq = args["train"]["checkpoint_freq"]
    store_path = os.path.join(args["results_dir"], "last.ckpt")
    criterion = getattr(torch.nn, args["train"]["criterion"])()
    optim = getattr(torch.optim, args["train"]["optimizer"])(
        model.get_trainable_parameters(),
        lr=args["train"]["lr"],
        maximize=(args["train"]["mode"] == "max"),
    )

    # Training loop
    accelerator = Accelerator()
    train_rtokens(
        model,
        train_loader,
        criterion,
        attack_fn,
        optim,
        accelerator,
        max_steps,
        checkpoint_freq,
        store_path,
    )

    # Loading latest model
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(store_path, map_location=accelerator.device))

    # Computing metrics on val set
    cossims, mses = evaluate_robustness(model, val_loader, attack_fn, accelerator)

    # Saving metrics
    cossims = pd.DataFrame.from_dict(cossims)
    mses = pd.DataFrame.from_dict(mses)
    cossims.to_csv(os.path.join(args["results_dir"], "cossims.csv"))
    mses.to_csv(os.path.join(args["results_dir"], "mses.csv"))

    wandb.log(dict(cossims.mean()))
    wandb.log(dict(mses.mean()))

    # Converting checkpoint
    torch.save(
        convert_checkpoint(model.cpu().state_dict()),
        os.path.join(args["results_dir"], "last.pth"),
    )

    # Finishing wandb
    wandb.finish()
    print("\n\n\nProgram completed successfully.")


if __name__ == "__main__":
    main(read_config())


if __name__ == "__main__":
    main()
