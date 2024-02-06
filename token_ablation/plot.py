import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

TOKENS = [1, 10, 20, 50]


def main():
    for file_name in ["tokens_small", "tokens_base", "tokens_large", "tokens_giant"]:
        size = file_name.split("_")[1]
        csv = pd.read_csv(f"{file_name}.csv")
        keys = [
            f"token_ablation - {size} - {n_tokens} - Train Loss" for n_tokens in TOKENS
        ]
        labels = [f"{n_tokens} tokens" for n_tokens in TOKENS]

        for k, l in zip(keys, labels):
            values = csv[k]
            plt.plot(np.arange(len(values)), values.to_numpy(), label=l)

        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Train loss")
        plt.ylim((0, 2.1))
        plt.savefig(f"{file_name}.png", dpi=500)
        plt.clf()


if __name__ == "__main__":
    main()
