"""DEIT-III model from https://github.com/facebookresearch/deit"""

import torch
from torch.nn import Module
import timm

SUPPORTED_DEIT3_MODELS = [
    "deit3_small_patch16_224",
    "deit3_base_patch16_224",
    "deit3_large_patch16_224",
    "deit3_huge_patch14_224",
]


def get_model(name, n_rtokens=1, enbable_robust=True):
    if name in SUPPORTED_DEIT3_MODELS:
        return DEIT3Robustifier(
            model_name=name,
            enable_robust=enbable_robust,
            n_rtokens=n_rtokens,
        )
    raise KeyError(f"Model {name} not supported. Pick one of {SUPPORTED_DEIT3_MODELS}")


class DEIT3Robustifier(Module):
    def __init__(self, model_name, n_rtokens=10, enable_robust=False):
        super(DEIT3Robustifier, self).__init__()

        assert (
            model_name in SUPPORTED_DEIT3_MODELS
        ), f"{model_name} not supported. Pick one of {SUPPORTED_DEIT3_MODELS}"

        self.model_name = model_name
        self.model = timm.create_model(
            "hf_hub:timm/" + model_name + ".fb_in1k", pretrained=True
        ).eval()
        self.enable_robust = enable_robust
        self.n_rtokens = max(0, n_rtokens)

        if self.n_rtokens > 0:
            hidden_dim = self.model.embed_dim
            self.rtokens = torch.nn.Parameter(
                1e-2 * torch.randn(1, n_rtokens, hidden_dim)
            )

    def get_trainable_parameters(self):
        if self.n_rtokens > 0:
            return [self.rtokens]
        return self.parameters()

    def store_rtokens(self, path=None):
        if self.n_rtokens > 0:
            if path is None:
                path = f"{self.model_name}_rtokens.pt"
            torch.save({"rtokens": self.rtokens}, path)

    def load_rtokens(self, path=None, device=None):
        if self.n_rtokens > 0:
            if path is None:
                path = f"{self.model_name}_rtokens.pt"
            self.rtokens = torch.load(path, map_location=device)["rtokens"]

    def forward(self, x, enable_robust=None, return_all=False):
        running_robust = (
            enable_robust if enable_robust is not None else self.enable_robust
        )

        return self.deit3_forward(x, running_robust, return_all)

    def deit3_forward(self, x, running_robust, return_all=False):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        # Appending robust tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(x.shape[0], 1, 1)), dim=1)

        x = self.model.blocks(x)

        rob_tokens = self.n_rtokens > 0 and running_robust
        x_norm = self.model.norm(x)
        cls_tokens = x_norm[:, 0]
        patch_tokens = x_norm[:, 1 : -self.n_rtokens] if rob_tokens else x_norm[:, 1:]
        rtokens = x_norm[:, -self.n_rtokens :] if rob_tokens else None

        # Returning all if requested
        if return_all:
            return {
                "x_norm_clstoken": cls_tokens,
                "x_norm_patchtokens": patch_tokens,
                "x_norm_rtokens": rtokens,
                "x_prenorm": x,
            }

        return torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)


if __name__ == "__main__":
    model_name = "deit3_small_patch16_224"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = (
        timm.create_model("hf_hub:timm/" + model_name + ".fb_in1k", pretrained=True)
        .eval()
        .to(device)
    )
    model = DEIT3Robustifier(model_name).eval().to(device)

    x = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        model.enable_robust = False
        y_original = original.forward_features(x)
        y = model(x)
        assert torch.allclose(y_original, y, atol=1e-6), f"y_original and y differ"

        model.enable_robust = True
        y2 = model(x)
        assert not torch.allclose(y, y2, atol=1e-6), f"y and y2 are the same"
        assert y.shape == y2.shape, f"y and y2 have different shapes"

    print("All tests passed!")
