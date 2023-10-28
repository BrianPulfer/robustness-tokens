import torch
from models.dinov2 import SUPPORTED_DINOV2_MODELS, get_model as get_dinov2_model


def test_dinov2():
    batch = torch.randn(1, 3, 224, 224).cuda()
    for name in SUPPORTED_DINOV2_MODELS:
        model = get_dinov2_model(name).eval().cuda()

        with torch.no_grad():
            f1 = model(batch, enable_robust=False, return_cls=True)
            f2 = model(batch, enable_robust=True, return_cls=True)
            f3 = model.model(batch)

            assert torch.allclose(
                f1, f3
            ), f"{name} failed: robustness tokens not disabled"
            assert not torch.allclose(
                f2, f3
            ), f"{name} failed: robustness tokens not enabled"
