import os
import random
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.nn.functional import mse_loss

import wandb
from attacks import pgd_attack
from data.imagenet import get_loaders as imagenet_loaders
from data.imagenette import get_loaders as imagenette_loaders
from data.transforms import unnormalize
from models import get_model


def validation_loop(model, loader, criterion, attack_fn, max_val_steps=None):
    loss_inv, loss_adv, total_mse = 0.0, 0.0, 0.0
    n_steps = len(loader) if max_val_steps is None else min(max_val_steps, len(loader))
    loader_iterator = iter(loader)

    for step in tqdm(range(n_steps), leave=False, desc="Validation loop"):
        batch = next(loader_iterator)
        batch = batch[0]
        batch_adv = attack_fn(model, batch)

        with torch.no_grad():
            target = model(batch, enable_robust=False)
            mse = mse_loss(unnormalize(batch_adv), unnormalize(batch))

            li = criterion(model(batch, enable_robust=True), target).mean()
            la = criterion(model(batch_adv, enable_robust=True), target).mean()

            loss_inv += li.item() * len(batch)
            loss_adv += la.item() * len(batch)
            total_mse += mse.item() * len(batch)

    return loss_inv, loss_adv, total_mse


def test_loop(model, loader, criterion, attack_fn):
    loss_inv_d, loss_adv_d = 0.0, 0.0
    loss_inv_r, loss_adv_r = 0.0, 0.0
    total_mse = 0.0
    for batch in tqdm(loader, leave=False, desc="Validation loop"):
        batch = batch[0]
        batch_adv = attack_fn(model, batch)

        with torch.no_grad():
            target = model(batch, enable_robust=False)
            mse = mse_loss(unnormalize(batch_adv), unnormalize(batch))

            # Default model
            lid = criterion(model(batch, enable_robust=False), target).mean()
            lad = criterion(model(batch_adv, enable_robust=False), target).mean()
            loss_inv_d += lid.item() * len(batch)
            loss_adv_d += lad.item() * len(batch)

            # With robustness tokens
            lir = criterion(model(batch, enable_robust=True), target).mean()
            lar = criterion(model(batch_adv, enable_robust=True), target).mean()
            loss_inv_r += lir.item() * len(batch)
            loss_adv_r += lar.item() * len(batch)

            # Image MSE
            total_mse += mse.item() * len(batch)
    return loss_inv_r, loss_adv_r, loss_inv_d, loss_adv_d, total_mse


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

    # Loading model robustifier
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

    model = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=args["model"]["return_cls"],
        n_rtokens=args["model"]["n_rtokens"],
    )

    wandb.watch(model, log="gradients", log_freq=args["train"]["grad_log_freq"])

    # Loading dataset
    if args["dataset"] == "imagenet":
        loaders_fn = imagenet_loaders
    elif args["dataset"] == "imagenette":
        loaders_fn = imagenette_loaders
    else:
        raise KeyError(f"Dataset {args['dataset']} not supported.")

    train_loader, val_loader = loaders_fn(
        args["train"]["batch_size"], args["train"]["num_workers"]
    )
    n_val_samples = len(val_loader.dataset)

    # Attacking model on dataset
    steps, best_val_loss = 0, float("inf")
    store_path = os.path.join(args["results_dir"], "best.ckpt")
    criterion = getattr(torch.nn, args["train"]["criterion"])()
    optim = getattr(torch.optim, args["train"]["optimizer"])(
        [model.rtokens],
        lr=args["train"]["lr"],
        maximize=(args["train"]["mode"] == "max"),
    )
    accelerator = Accelerator()
    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    with tqdm(total=args["train"]["max_steps"], desc="Training") as pbar:
        while steps < args["train"]["max_steps"]:
            for batch in train_loader:
                batch = batch[0]
                batch_adv = attack_fn(model, batch)
                with torch.no_grad():
                    target = model(batch, enable_robust=False)
                    mse = mse_loss(unnormalize(batch_adv), unnormalize(batch))

                for _ in range(args["train"]["steps_per_batch"]):
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

                if steps % args["train"]["val_every_n_steps"] == 0:
                    torch.cuda.empty_cache()
                    l_inv, l_adv, mse = validation_loop(
                        model,
                        val_loader,
                        criterion,
                        attack_fn,
                        args["train"]["max_val_steps"],
                    )
                    val_loss = l_inv + l_adv
                    wandb.log(
                        {
                            "Val Loss": (l_inv + l_adv) / n_val_samples,
                            "Val Loss Invariance": l_inv / n_val_samples,
                            "Val Loss Adversarial": l_adv / n_val_samples,
                            "Val Image MSE": mse,
                        },
                        step=steps,
                    )
                    if val_loss < best_val_loss:
                        torch.save(accelerator.get_state_dict(model), store_path)

                steps += 1
                pbar.update(1)
                if steps >= args["train"]["max_steps"]:
                    break

    # Testing model with and without robustification
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(store_path, map_location=accelerator.device))
    l_inv_r, l_adv_r, l_inv_d, l_adv_d, mse = test_loop(
        model, val_loader, criterion, attack_fn
    )
    wandb.log(
        {
            "Final val Loss (with robustness tokens)": (l_inv_r + l_adv_r)
            / n_val_samples,
            "Final val Loss Invariance (with robustness tokens)": l_inv_r
            / n_val_samples,
            "Final val Loss Adversarial (with robustness tokens)": l_adv_r
            / n_val_samples,
            "Final val Loss (without robustness tokens)": (l_inv_d + l_adv_d)
            / n_val_samples,
            "Final val Loss Invariance (without robustness tokens)": l_inv_d
            / n_val_samples,
            "Final val Loss Adversarial (without robustness tokens)": l_adv_d
            / n_val_samples,
            "Final val Image MSE (with robustness tokens)": mse,
        },
        step=steps,
    )

    # Finishing wandb
    wandb.finish()
    print("\n\n\nProgram completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = vars(parser.parse_args())

    if args["config"] is not None:
        if os.path.isfile(args["config"]):
            with open(args["config"], "r") as f:
                args = yaml.safe_load(f)
        else:
            warnings.warn(f"Config file {args['config']} not found.")
            exit()

    print("\n\nProgram arguments:\n", args)
    main(args)
