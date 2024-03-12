from argparse import ArgumentParser

import os
import types

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Normalize
from torchvision.utils import save_image

import timm


class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, extensions=[".png"]):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, img)
            for img in os.listdir(root_dir)
            if any([img.endswith(ext) for ext in extensions])
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image


def attack(
    model,
    batch,
    validate_fn,
    unnormalize_fn,
    lr=4e-4,
    steps=350,
    eps=8 / 255,
    target_mse=1e-4,
):
    """Attacks the batch using adversarial attack."""

    # Updating eps to be in the same scale as the features
    eps = eps * (batch.max() - batch.min())

    device = batch.device
    batch_adv = (torch.randn_like(batch) * lr + batch).detach().clone()
    cossim = torch.nn.CosineSimilarity(dim=1).to(device)

    batch_img = unnormalize_fn(batch)

    with torch.no_grad():
        label = model.forward_features(batch)

    best_loss, best_adv = float("inf"), batch_adv
    for _ in tqdm(range(steps), desc="Computing attack..."):
        batch_adv.requires_grad = True

        l_feat = cossim(model.forward_features(batch_adv), label).mean()
        l_img = (batch_img - unnormalize_fn(batch_adv)).pow(2).mean()
        loss = l_feat  # + l_img  / target_mse
        grad = torch.autograd.grad(loss, batch_adv)[0]

        print(
            f"Loss: {loss.item():.5f}, MSE: {l_img.item():.5f}, CosSim: {l_feat.item():.5f}"
        )

        if l_img >= target_mse:
            print("Exceeded target MSE")
            break

        if loss.item() < best_loss:
            best_loss, best_adv = loss.item(), batch_adv.clone()

        delta = torch.clamp(batch_adv - lr * grad.sign() - batch, -eps, eps)
        batch_new = batch + delta
        if torch.all(batch_new == batch_adv):
            print("Same as before")
            break
        batch_adv = batch_new.detach()
    return validate_fn(best_adv).to(device)


def get_model(model_family, model_size):
    if model_family == "mae":
        patch_size = 14 if model_size == "huge" else 16
        model = timm.create_model(
            f"vit_{model_size}_patch{patch_size}_224.mae", pretrained=True
        )
    elif model_family == "openai_clip":
        patch_size = 14 if model_size == "large" else 16
        model = timm.create_model(
            f"vit_{model_size}_patch{patch_size}_clip_224.openai", pretrained=True
        )
    elif model_family == "dinov2":
        model = timm.create_model(
            f"vit_{model_size}_patch14_dinov2.lvd142m",
            pretrained=True,
        )
    elif model_family == "dinov2_reg":
        model = timm.create_model(
            f"vit_{model_size}_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
        )

    model = model.cuda()
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    to_pil = ToPILImage()
    normalize = val_transform.transforms[-1]
    unnormalize = Normalize(
        mean=[-m / s for m, s in zip(normalize.mean, normalize.std)],
        std=[1 / s for s in normalize.std],
    )

    def validate_fn(x):
        x = unnormalize(x).clamp(0, 1)
        imgs = [to_pil(img) for img in x]
        x = torch.stack([val_transform(img) for img in imgs])
        return x

    return model, val_transform, validate_fn, unnormalize


def change_output(model, model_name, layer):
    """Changes the model such that the output is the output of the layer-th layer."""
    assert "dinov2" in model_name or "mae" in model_name, "Model not supported"

    def vit_custom_block_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        self.feat = x.clone().detach().cpu().double()
        return x

    model.blocks[layer].forward = types.MethodType(
        vit_custom_block_forward, model.blocks[layer]
    )


def store_feat_plot(
    dataset,
    n_images,
    validate_fn,
    model,
    model_family,
    model_size,
    unnormalize_fn,
    layer,
    savedir,
    device=None,
):
    """Computes and stores the plot of the features"""

    # Getting original "batch"
    idxs = np.random.choice(len(dataset), n_images, replace=False)
    batch = torch.stack([dataset[i] for i in idxs]).to(device)

    # Computing adversarial batch
    adv_batch = attack(model, batch, validate_fn, unnormalize_fn)
    save_image(unnormalize_fn(adv_batch[0]), "debug.png")

    # Changing the output of the model to be the output of the layer
    change_output(model, model_family, layer)

    # Getting the features
    with torch.no_grad():
        out = model.forward_features(batch)
        feat = model.blocks[layer].feat
        adv_out = model.forward_features(adv_batch)
        adv_feat = model.blocks[layer].feat

        print("\nMean squared error between original and adversarial images:")
        print((unnormalize_fn(batch) - unnormalize_fn(adv_batch)).pow(2).mean().item())

        print("\nMean cosine similarity between original and adversarial output:")
        print(torch.nn.functional.cosine_similarity(out, adv_out).mean().item())

        print(
            f"\nMean cosine similarity between original and adversarial features at layer {layer}:"
        )
        print(torch.nn.functional.cosine_similarity(feat, adv_feat).mean().item())

        feat = feat.abs()
        adv_feat = adv_feat.abs()

    # Plotting features in 3D
    # x-axis: token index
    # y-axis: token dimensions
    # z-axis: magnitude
    X, Y = np.meshgrid(np.arange(feat.shape[-2]), np.arange(feat.shape[-1]))
    for f, name, color in zip(
        [feat, adv_feat], ["original", "adversarial"], ["royalblue", "firebrick"]
    ):
        fig = plt.figure(figsize=(8, 6))
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        plt.subplots_adjust(wspace=0.0)

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        Z = f[0].abs().cpu().numpy().T
        ax.plot_wireframe(X, Y, Z, rstride=0, color=color, linewidth=2.5)
        plt.savefig(os.path.join(savedir, f"feat_{layer}_{name}.png"))


def main():
    # Arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--model_family", type=str, default="dinov2", help="Model family"
    )
    parser.add_argument("--model_size", type=str, default="large", help="Model size")
    parser.add_argument("--layer", type=int, default=22, help="Layer to visualize")
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=1,
        help="Number of images to visualize features for",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/coco2017",
        help="Folder with images to visualize features",
    )
    parser.add_argument(
        "--savedir", type=str, default="plots", help="Directory to save the plots"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = vars(parser.parse_args())

    seed = args["seed"]
    model_family = args["model_family"].lower()
    model_size = args["model_size"].lower()
    layer = args["layer"]
    n_imgs = args["num_imgs"]
    root_dir = args["image_folder"]
    savedir = args["savedir"]

    # Setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Creating directory to save plots
    os.makedirs(savedir, exist_ok=True)

    # Loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, validate_fn, unnormalize_fn = get_model(model_family, model_size)
    model = model.eval().to(device)

    # Loading data
    dataset = ImageFolder(root_dir, transform=transform)

    # Visualizing features
    store_feat_plot(
        dataset,
        n_imgs,
        validate_fn,
        model,
        model_family,
        model_size,
        unnormalize_fn,
        layer,
        savedir,
        device,
    )


if __name__ == "__main__":
    main()
