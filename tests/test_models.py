import torch

from models.dinov2 import SUPPORTED_DINOV2_MODELS, get_model as get_dinov2_model


def test_dinov2():
    """Tests that the DINOv2 models are correctly loaded and return the same output when robustness tokens are disabled."""
    batch = torch.randn(1, 3, 224, 224).cuda()
    for name in SUPPORTED_DINOV2_MODELS:
        wrapper = get_dinov2_model(name).eval().cuda()
        dino = wrapper.model.eval().cuda()

        with torch.no_grad():
            dino_out = dino.forward_features(batch)
            wrapper_out = wrapper(batch, enable_robust=False, return_all=True)

            # Testing that CLS token stays the same
            assert torch.allclose(
                dino_out["x_norm_clstoken"],
                wrapper_out["x_norm_clstoken"],
                atol=1e-5,
            ), f"CLS token mismatch for {name}"

            # Testing that patch tokens stay the same
            assert torch.allclose(
                dino_out["x_norm_patchtokens"],
                wrapper_out["x_norm_patchtokens"],
                atol=1e-5,
            ), f"Patch tokens mismatch for {name}"

            if "_reg" in name:
                # Testing that register tokens stay the same
                assert torch.allclose(
                    dino_out["x_norm_regtokens"],
                    wrapper_out["x_norm_regtokens"],
                    atol=1e-5,
                ), f"Register tokens mismatch for {name}"
