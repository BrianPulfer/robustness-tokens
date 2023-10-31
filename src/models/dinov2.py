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
            # self.rtokens = torch.nn.Parameter(torch.zeros(1, n_rtokens, hidden_dim))
        else:
            self.extra_params = torch.nn.ParameterList(
                [1e-3 * torch.randn_like(p) for p in self.model.parameters()]
            )
            self.added_params = False

    def get_trainable_parameters(self):
        if self.n_rtokens > 0:
            return [self.rtokens]
        return self.extra_params

    def forward(self, x, enable_robust=None, return_cls=None, return_rtokens=False):
        b, c, w, h = x.shape
        running_cls = return_cls if return_cls is not None else self.return_cls
        running_robust = (
            enable_robust if enable_robust is not None else self.enable_robust
        )

        if self.n_rtokens == 0:
            if running_robust:
                self.add_extra_params()
            else:
                self.remove_extra_params()

        # Embedding patches
        x = self.model.patch_embed(x)

        # Concatenating class token
        x = torch.cat((self.model.cls_token.expand(b, -1, -1), x), dim=1)
        x += self.model.interpolate_pos_encoding(x, w, h)

        # Appending robust tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(b, 1, 1)), dim=1)

        # Running blocks
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)

        if running_cls:
            return self.model.head(x[:, 0])

        if not return_rtokens and self.n_rtokens > 0 and running_robust:
            return x[:, : -self.n_rtokens]

        return x

    def add_extra_params(self):
        if not self.added_params:
            for p1, p2 in zip(self.extra_params, self.model.parameters()):
                p2.data += p1.data
            self.added_params = True

    def remove_extra_params(self):
        if self.added_params:
            for p1, p2 in zip(self.extra_params, self.model.parameters()):
                p2.data -= p1.data
            self.added_params = False
