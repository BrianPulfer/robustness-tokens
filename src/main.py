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
from data.imagenette import get_loaders
from data.transforms import unnormalize
from models import get_model


def validation_loop(model, loader, criterion, attack_fn):
    loss_inv, loss_adv, total_mse = 0.0, 0.0, 0.0
    for batch in tqdm(loader, leave=False, desc="Validation loop"):
        batch_adv = attack_fn(model, batch)

        with torch.no_grad():
            target = model(batch, enable_robust=False)
            mse = mse_loss(unnormalize(batch_adv), unnormalize(batch))

            li = criterion(model(batch, enable_robust=True), target).mean()
            la = criterion(model(batch_adv, enable_robust=True), target).mean()

            loss_inv += li.item() * len(batch)
            loss_adv += la.item() * len(batch)
            total_mse += mse.item() * len(batch)
    return loss_inv, loss_inv, total_mse


def test_loop(model, loader, criterion, attack_fn):
    loss_inv_d, loss_adv_d = 0.0, 0.0
    loss_inv_r, loss_adv_r = 0.0, 0.0
    total_mse = 0.0
    for batch in tqdm(loader, leave=False, desc="Validation loop"):
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

    # Creating result directory
    os.makedirs(args["results_dir"], exist_ok=True)

    # Loading model robustifier
    def attack_fn(model, batch):
        return pgd_attack(
            model,
            batch,
            steps=args["attack"]["steps"],
            lr=args["attack"]["lr"],
            eps=args["attack"]["eps"],
            max_mse=args["attack"]["max_mse"],
        )

    model = get_model(
        args["model"]["name"],
        enbable_robust=True,
        return_cls=args["model"]["return_cls"],
        n_rtokens=args["model"]["n_rtokens"],
    )

    wandb.watch(model, log="gradients", log_freq=args["train"]["grad_log_freq"])

    # Loading dataset
    # TODO: Use ImageNet dataset instead
    train_loader, val_loader, test_loader = get_loaders(
        args["train"]["batch_size"], args["train"]["num_workers"]
    )
    n_val_samples = len(val_loader.dataset)
    n_test_samples = len(test_loader.dataset)

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
    model, optim, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optim, train_loader, val_loader, test_loader
    )

    with tqdm(total=args["train"]["max_steps"], desc="Training") as pbar:
        while steps < args["train"]["max_steps"]:
            for batch in train_loader:
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
                        "Train Loss Adversarial": loss_inv.item(),
                        "Train Loss Invariance": loss_adv.item(),
                    },
                    step=steps,
                )

                if steps % args["train"]["val_every_n_steps"] == 0:
                    torch.cuda.empty_cache()
                    l_inv, l_adv, mse = validation_loop(
                        model, val_loader, criterion, attack_fn
                    )
                    val_loss = l_inv + l_adv
                    wandb.log(
                        {
                            "Val Loss": (l_inv + l_adv) / n_val_samples,
                            "Val Loss Adversarial": l_inv / n_val_samples,
                            "Val Loss Invariance": l_adv / n_val_samples,
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
        model, test_loader, criterion, attack_fn
    )
    wandb.log(
        {
            "Test loss (with robustness tokens)": (l_inv_r + l_adv_r) / n_test_samples,
            "Test Loss Invariance (with robustness tokens)": l_inv_r / n_test_samples,
            "Test Loss Adversarial (with robustness tokens)": l_adv_r / n_test_samples,
            "Test loss (without robustness tokens)": (l_inv_d + l_adv_d)
            / n_test_samples,
            "Test Loss Invariance (without robustness tokens)": l_inv_d
            / n_test_samples,
            "Test Loss Adversarial (without robustness tokens)": l_adv_d
            / n_test_samples,
            "Test Image MSE (with robustness tokens)": mse,
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
