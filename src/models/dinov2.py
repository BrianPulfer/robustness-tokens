from copy import deepcopy

import torch
from torch.nn import Module

SUPPORTED_DINOV2_MODELS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]


def get_model(name, enbable_robust=True, return_cls=False, n_rtokens=1):
    if name in SUPPORTED_DINOV2_MODELS:
        return DinoV2Robustifier(
            model_name=name,
            enable_robust=enbable_robust,
            return_cls=return_cls,
            n_rtokens=n_rtokens,
        )
    raise KeyError(f"Model {name} not supported. Pick one of {SUPPORTED_DINOV2_MODELS}")


class DinoV2Robustifier(Module):
    def __init__(self, model_name, enable_robust=False, return_cls=False, n_rtokens=1):
        super(DinoV2Robustifier, self).__init__()

        assert (
            model_name in SUPPORTED_DINOV2_MODELS
        ), f"{model_name} not supported. Pick one of {SUPPORTED_DINOV2_MODELS}"

        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).eval()
        self.enable_robust = enable_robust
        self.return_cls = return_cls
        self.n_rtokens = max(0, n_rtokens)

        if self.n_rtokens > 0:
            hidden_dim = self.model.cls_token.shape[-1]
            self.rtokens = torch.nn.Parameter(
                1e-2 * torch.randn(1, n_rtokens, hidden_dim)
            )
        else:
            self.model_copy = deepcopy(self.model)
            self.model.requires_grad_(False)

    def get_trainable_parameters(self):
        if self.n_rtokens > 0:
            return [self.rtokens]
        return self.model_copy.parameters()

    def forward(self, x, enable_robust=None, return_cls=None, return_rtokens=False):
        running_cls = return_cls if return_cls is not None else self.return_cls
        running_robust = (
            enable_robust if enable_robust is not None else self.enable_robust
        )

        h = self.dino_forward(self.model, x, running_robust)

        if self.n_rtokens == 0:
            if running_robust:
                return h + self.dino_forward(self.model_copy, x, running_robust)

        if running_cls:
            return self.model.head(h[:, 0])

        if not return_rtokens and self.n_rtokens > 0 and running_robust:
            return h[:, : -self.n_rtokens]

        return h

    def dino_forward(self, model, x, running_robust):
        b, c, w, h = x.shape

        # Embedding patches
        x = model.patch_embed(x)

        # Concatenating class token
        x = torch.cat((model.cls_token.expand(b, -1, -1), x), dim=1)
        x += model.interpolate_pos_encoding(x, w, h)

        # Appending robust tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(b, 1, 1)), dim=1)

        # Running blocks
        for blk in self.model.blocks:
            x = blk(x)
        x = model.norm(x)
        return x
