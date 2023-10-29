import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

import wandb
from attacks import pgd_attack
from data.transforms import unnormalize
from data.utils import get_loaders_fn
from eval.robustness import evaluate_rtokens
from models.utils import get_model
from utils import read_config


def training_loop(
    model,
    loader,
    criterion,
    attack_fn,
    optim,
    accelerator,
    max_steps,
    steps_per_batch,
    checkpoint_freq,
    store_path,
):
    """Training loop to optimize robustness tokens."""
    # Preparing model, optimizer and data loader
    model, optim, loader = accelerator.prepare(model, optim, loader)

    # Loop
    steps = 0
    with tqdm(total=max_steps, desc="Training") as pbar:
        while steps < max_steps:
            for batch in loader:
                batch = batch[0]
                batch_adv = attack_fn(model, batch)
                with torch.no_grad():
                    target = model(batch, enable_robust=False)
                    mse = mse_loss(unnormalize(batch_adv), unnormalize(batch))

                for _ in range(steps_per_batch):
                    loss_inv = criterion(
                        model(batch, enable_robust=True), target
                    ).mean()
                    loss_adv = criterion(
                        model(batch_adv, enable_robust=True), target
                    ).mean()
                    loss = loss_inv + loss_adv
                    optim.zero_grad()
                    accelerator.backward(loss)
                    optim.step()

                wandb.log(
                    {
                        "Train Loss": loss.item(),
                        "Train Image MSE": mse.item(),
                        "Train Loss Invariance": loss_inv.item(),
                        "Train Loss Adversarial": loss_adv.item(),
                    },
                    step=steps,
                )

                steps += 1
                pbar.update(1)

                if steps % checkpoint_freq == 0:
                    torch.save(accelerator.get_state_dict(model), store_path)

                if steps >= max_steps:
                    break
    torch.save(accelerator.get_state_dict(model), store_path)


def main(args):
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

    # Defining attack function
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

    # Initializing model
    model = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=args["model"]["return_cls"],
        n_rtokens=args["model"]["n_rtokens"],
    )
    wandb.watch(model, log="gradients", log_freq=args["train"]["grad_log_freq"])

    # Preparing data loaders
    loaders_fn = get_loaders_fn(args["dataset"])

    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"], num_workers=args["train"]["num_workers"]
    )

    # Training hyper-parameters
    max_steps = args["train"]["max_steps"]
    steps_per_batch = args["train"]["steps_per_batch"]
    checkpoint_freq = args["train"]["checkpoint_freq"]
    store_path = os.path.join(args["results_dir"], "last.ckpt")
    criterion = getattr(torch.nn, args["train"]["criterion"])()
    optim = getattr(torch.optim, args["train"]["optimizer"])(
        [model.rtokens],
        lr=args["train"]["lr"],
        maximize=(args["train"]["mode"] == "max"),
    )

    # Training loop
    accelerator = Accelerator()
    training_loop(
        model,
        train_loader,
        criterion,
        attack_fn,
        optim,
        accelerator,
        max_steps,
        steps_per_batch,
        checkpoint_freq,
        store_path,
    )

    # Loading latest model
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(store_path, map_location=accelerator.device))

    # Computing metrics on val set
    cossims, mses = evaluate_rtokens(model, val_loader, attack_fn, accelerator)

    # Saving metrics
    cossims = pd.DataFrame.from_dict(cossims)
    mses = pd.DataFrame.from_dict(mses)
    cossims.to_csv(os.path.join(args["results_dir"], "cossims.csv"))
    mses.to_csv(os.path.join(args["results_dir"], "mses.csv"))

    # wandb.log(
    #    {"cossims": wandb.Table(dataframe=cossims), "mses": wandb.Table(dataframe=mses)}
    # )

    # Finishing wandb
    wandb.finish()
    print("\n\n\nProgram completed successfully.")


if __name__ == "__main__":
    main(read_config())
