import torch
from torch.nn import Module

SUPPORTED_DINOV2_MODELS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]


def get_model(name, n_rtokens=10, enbable_robust=True):
    if name in SUPPORTED_DINOV2_MODELS:
        return DinoV2Robustifier(
            model_name=name,
            enable_robust=enbable_robust,
            n_rtokens=n_rtokens,
        )
    raise KeyError(f"Model {name} not supported. Pick one of {SUPPORTED_DINOV2_MODELS}")


class DinoV2Robustifier(Module):
    def __init__(self, model_name, n_rtokens=10, enable_robust=False):
        super(DinoV2Robustifier, self).__init__()

        assert (
            model_name in SUPPORTED_DINOV2_MODELS
        ), f"{model_name} not supported. Pick one of {SUPPORTED_DINOV2_MODELS}"

        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).eval()
        self.enable_robust = enable_robust
        self.n_rtokens = max(0, n_rtokens)

        if self.n_rtokens > 0:
            hidden_dim = self.model.cls_token.shape[-1]
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

        return self.dino_forward(x, running_robust, return_all)

    def dino_forward(self, x, running_robust, return_all=False):
        b, c, w, h = x.shape

        # Embedding patches
        x = self.model.patch_embed(x)

        # Concatenating class token + positional encoding
        x = torch.cat((self.model.cls_token.expand(b, -1, -1), x), dim=1)
        x += self.model.interpolate_pos_encoding(x, w, h)

        # Concatenating registers
        if self.model.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.model.register_tokens.expand(b, -1, -1), x[:, 1:]),
                dim=1,
            )

        # Appending robust tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(b, 1, 1)), dim=1)

        # Running blocks
        for blk in self.model.blocks:
            x = blk(x)

        x_norm = self.model.norm(x)

        # Counting the number of tokens used
        regt = (
            self.model.num_register_tokens
            if self.model.register_tokens is not None
            else 0
        )
        robt = self.n_rtokens if running_robust else 0

        cls_tokens = x_norm[:, 0]
        patch_tokens = x_norm[:, regt + 1 : (None if robt == 0 else -robt)]

        if return_all:
            return {
                "x_norm_clstoken": cls_tokens,
                "x_norm_regtokens": x_norm[:, 1 : regt + 1] if regt > 0 else None,
                "x_norm_patchtokens": patch_tokens,
                "x_norm_rtokens": x_norm[:, -robt:] if robt > 0 else None,
                "x_prenorm": x,
            }

        return torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)
