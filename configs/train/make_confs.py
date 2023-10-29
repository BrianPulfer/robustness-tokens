import yaml

model_batches_steps = [
    ("dinov2_vits14", 256, 300),
    ("dinov2_vitb14", 128, 300),
    ("dinov2_vitl14", 64, 300),
]


def main():
    for model_name, batch_size, steps in model_batches_steps:
        for ntokens in [1, 2, 10, 30]:
            d = {
                "attack": {"eps": 0.0314, "lr": 0.01, "max_mse": 0.0001, "steps": 30},
                "dataset": "imagenet",
                "model": {
                    "n_rtokens": ntokens,
                    "name": model_name,
                    "return_cls": False,
                },
                "results_dir": f"results/{model_name}_{ntokens}tokens",
                "run_name": f"{model_name} | {ntokens} RTokens",
                "seed": 0,
                "train": {
                    "batch_size": 128,
                    "checkpoint_freq": 30,
                    "criterion": "CosineSimilarity",
                    "lr": 0.001,
                    "max_steps": steps,
                    "mode": "max",
                    "num_workers": 16,
                    "optimizer": "Adam",
                    "steps_per_batch": 1,
                },
            }

            with open(f"configs/train/{model_name}_{ntokens}tokens.yaml", "w") as f:
                yaml.dump(d, f)


if __name__ == "__main__":
    main()
