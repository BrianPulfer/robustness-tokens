from argparse import ArgumentParser


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# plt.style.use(["science", "grid"])
plt.style.use("science")


def main(args):
    # Read the file
    df = pd.read_csv(args["file"])

    # Only keeping the columns which contain the 'Train Loss' substring
    df = df[df.columns[df.columns.str.endswith(args['column_end'])]]

    # Plot the train curves for all the columns
    for column in sorted(df.columns):
        data = df[column]
        label = column.split(" ")[0]
        label = " ".join(label.split("_")[:2])
        label = label.replace("vitb16", "base")
        label = label.replace("vith14", "huge")
        label = label.replace("vitl14", "large")
        plt.plot(np.arange(len(data)), data, label=label)

    plt.xlabel("Step")
    plt.ylabel(args['ylabel'])
    plt.title(args["title"], fontdict={"fontsize": 20})
    plt.ylim(args['ymin'], args['ymax'])
    plt.legend()
    plt.savefig(args['filename'], dpi=300)

if __name__ == "__main__":
    # Parse arguments. There's one "--file" of type str. Print args nicely and pass a dictionary of args to the main function
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--column_end", type=str, required=True, default="Train Loss")
    parser.add_argument("--ymin", type=float, required=True, default=0)
    parser.add_argument("--ymax", type=float, required=True, default=0.5)
    parser.add_argument("--ylabel", type=str, required=True, default="$\mathcal{L}_$")
    parser.add_argument("--title", type=str, required=True, default="")
    parser.add_argument("--filename", type=str, required=True, default="figure.png")
    args = vars(parser.parse_args())

    print(args)
    main(args)


### Commands ###
# L = Linv + Ladv
# python train_curves.py --file deit_openclip_trainloss.csv --title "" --column_end "Train Loss" --ymin 0 --ymax 2.1 --ylabel "$\mathcal{L} = \mathcal{L}_\\text{inv} + \mathcal{L}_\\text{adv}$" --filename train.png

# Linv
# python train_curves.py --file deit_openclip_invloss.csv --title "" --column_end "Train Loss Invariance" --ymin 0 --ymax 1.1 --ylabel "$\mathcal{L}_\\text{inv}$" --filename inv.png

# Ladv
# python train_curves.py --file deit_openclip_advloss.csv --title "" --column_end "Train Loss Adversarial" --ymin 0 --ymax 1.1 --ylabel "$\mathcal{L}_\\text{adv}$" --filename adv.png
### Commands ###