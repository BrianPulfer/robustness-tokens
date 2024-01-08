import os
import random

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

import wandb
from attacks.utils import get_attack
from data.transforms import unnormalize
from data.utils import get_loaders_fn
from models.utils import get_model
from utils import read_config


def train_rtokens(
    model,
    loader,
    criterion,
    attack_fn,
    optim,
    accelerator,
    max_steps,
    checkpoint_freq,
    store_path,
):
    """Training loop to optimize robustness tokens."""
    # Preparing model, optimizer and data loader
    model, optim, loader = accelerator.prepare(model, optim, loader)
    model_path = os.path.join(store_path, "last.ckpt")
    tokens_path = os.path.join(store_path, "rtokens.pt")

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
                    baseline = (
                        criterion(model(batch_adv, enable_robust=False), target)
                        .mean()
                        .item()
                    )

                loss_inv = criterion(model(batch, enable_robust=True), target).mean()
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
                        "Without robustness": baseline,
                    },
                    step=steps,
                )

                steps += 1
                pbar.update(1)

                if steps % checkpoint_freq == 0:
                    torch.save(accelerator.get_state_dict(model), model_path)
                    model.store_rtokens(tokens_path)

                if steps >= max_steps:
                    break
    torch.save(accelerator.get_state_dict(model), model_path)
    model.store_rtokens(tokens_path)


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

    # Initializing model
    model = get_model(**args["model"])

    # Defining attack function
    attack_fn = get_attack(model, **args["attack"])

    # Preparing data loaders
    loaders_fn = get_loaders_fn(args["dataset"])
    train_loader, _ = loaders_fn(
        args["train"]["batch_size"], num_workers=args["train"]["num_workers"]
    )

    # Training hyper-parameters
    max_steps = args["train"]["max_steps"]
    checkpoint_freq = args["train"]["checkpoint_freq"]
    store_path = args["results_dir"]
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

    # Finishing wandb
    wandb.finish()
    print("\n\n\nProgram completed successfully.")


if __name__ == "__main__":
    main(read_config())
