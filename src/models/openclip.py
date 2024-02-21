"""Open-CLIP model from https://github.com/mlfoundations/open_clip"""

import torch
from torch.nn import Module
import open_clip

SUPPORTED_OPENCLIP_MODELS = [
    "openclip_vitb32",
]

NAME_TO_ARGS = {
    "openclip_vitb32": {
        "model_name": "ViT-B/32",
        "pretrained": "laion2b_s34b_b79k",
    }
}


def get_model(name, n_rtokens=1, enbable_robust=True):
    if name in SUPPORTED_OPENCLIP_MODELS:
        return OpenCLIPRobustifier(
            model_name=name,
            enable_robust=enbable_robust,
            n_rtokens=n_rtokens,
        )
    raise KeyError(
        f"Model {name} not supported. Pick one of {SUPPORTED_OPENCLIP_MODELS}"
    )


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class OpenCLIPRobustifier(Module):
    def __init__(self, model_name, n_rtokens=10, enable_robust=False):
        super(OpenCLIPRobustifier, self).__init__()

        assert (
            model_name in SUPPORTED_OPENCLIP_MODELS
        ), f"{model_name} not supported. Pick one of {SUPPORTED_OPENCLIP_MODELS}"

        self.model_name = model_name
        self.model = open_clip.create_model(**NAME_TO_ARGS[model_name]).visual.eval()
        self.enable_robust = enable_robust
        self.n_rtokens = max(0, n_rtokens)

        if self.n_rtokens > 0:
            hidden_dim = self.model.class_embedding.shape[-1]
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

        return self.openclip_forward(x, running_robust, return_all)

    def openclip_forward(self, x, running_robust, return_all=False):
        b, c, w, h = x.shape

        # Open-CLIP pre-transformer steps
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [_expand_token(self.model.class_embedding, x.shape[0]).to(x.dtype), x],
            dim=1,
        )
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.patch_dropout(x)
        x = self.model.ln_pre(x)

        # Appending robust tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(b, 1, 1)), dim=1)

        # Open-CLIP transformer step
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Removing robustness tokens
        x = x[:, : -self.n_rtokens] if running_robust and self.n_rtokens > 0 else x

        # Open-CLIP post-transformer steps
        if self.model.attn_pool is not None:
            if self.model.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.model.attn_pool(x)
                if self.model.attn_pool_type == "parallel":
                    pooled = self.model.attn_pool_contrastive(x)
                else:
                    assert self.model.attn_pool_type == "cascade"
                    pooled = self.model.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.model.attn_pool(x)
                x = self.model.ln_post(x)
                pooled, tokens = self.model._global_pool(x)
        elif self.model.final_ln_after_pool:
            pooled, tokens = self.model._global_pool(x)
            pooled = self.model.ln_post(pooled)
        else:
            x = self.model.ln_post(x)
            pooled, tokens = self.model._global_pool(x)

        if self.model.proj is not None:
            pooled = pooled @ self.model.proj

        if self.model.output_tokens:
            return pooled, tokens

        return pooled


if __name__ == "__main__":
    model_name = "openclip_vitb32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = (
        open_clip.create_model(**NAME_TO_ARGS[model_name]).visual.eval().to(device)
    )
    model = OpenCLIPRobustifier(model_name).eval().to(device)

    x = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        model.enable_robust = False
        y_original = original(x)
        y = model(x)
        assert torch.allclose(y_original, y, atol=1e-6), f"y_original and y differ"

        model.enable_robust = True
        y2 = model(x)
        assert not torch.allclose(y, y2, atol=1e-6), f"y and y2 are the same"
        assert y.shape == y2.shape, f"y and y2 have different shapes"

    print("All tests passed!")
