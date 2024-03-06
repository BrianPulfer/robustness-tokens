import os
import types
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

# Changing color cycle
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "green",
        "red"
    ]
)

import torch
from accelerate import Accelerator

from utils import read_config
from models.utils import get_model
from attacks.pgd import pgd_attack
from data.transforms import unnormalize
from data.utils import get_loaders_fn


def main(args):
    # Setting seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])

    # Creating output directory
    savedir = args["savedir"]
    os.makedirs(savedir, exist_ok=True)

    # Loading model
    model_name = args["model"]["name"]
    model = get_model(**args["model"])

    if args.get("rtokens", None) is not None:
        model.load_rtokens(args["rtokens"])

    # Loading data
    loaders_fn = get_loaders_fn(args["dataset"])
    loader, _ = loaders_fn(args["batch_size"], num_workers=args["num_workers"])

    # Moving to device
    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)

    # Getting first adversarial batch
    for batch in loader:
        batch = batch[0]

        # Getting adversarial batch w.r.t. standard model
        model.enable_robust = False
        batch_adv = pgd_attack(model, batch)
        break

    # Layer indices
    idxs = args["layer_idxs"]

    # Extracting activations
    with torch.no_grad():
        for run_robust in [False, True]:
            model.enable_robust = run_robust

            # Activations are (Layers, Batch, Seq. len, Dim)
            activations = torch.stack(model(batch, return_layers=idxs))
            activations_adv = torch.stack(model(batch_adv, return_layers=idxs))

            # Permuting activations to (Batch, Layers, Seq. len, Dim)
            activations = activations.permute(1, 0, 2, 3)
            activations_adv = activations_adv.permute(1, 0, 2, 3)
            
            # Optionally removing robustness tokens
            activations = activations[:, :, :-model.n_rtokens]
            activations_adv = activations_adv[:, :, :-model.n_rtokens]

            # Getting the max abs value for each layer and sample
            activations = activations.abs().max(dim=-1).values.max(dim=-1).values
            activations_adv = (
                activations_adv.abs().max(dim=-1).values.max(dim=-1).values
            )

            # Plotting and saving activations
            for i, (act, act_adv) in enumerate(zip(activations, activations_adv)):
                plt.plot(act, '-', label="Standard") # , alpha=0.5)
                plt.plot(act_adv, '-', label="Adversarial") # , alpha=0.5)
                plt.xlabel("Layer")
                plt.ylabel("Max abs. value")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        savedir,
                        f"{model_name}_activations_{i}_{'robust' if run_robust else 'standard'}.png",
                    ),
                    dpi=1000,
                )
                plt.close()

    print("Done.")


if __name__ == "__main__":
    # Parse arguments
    main(read_config())
