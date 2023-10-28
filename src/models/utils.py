from models.dinov2 import SUPPORTED_DINOV2_MODELS, get_model as get_dinov2_model

ALL_SUPPORTED_MODELS = SUPPORTED_DINOV2_MODELS


def get_model(model_name, **kwargs):
    if model_name not in ALL_SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {ALL_SUPPORTED_MODELS}"
        )

    if model_name in SUPPORTED_DINOV2_MODELS:
        return get_dinov2_model(model_name, **kwargs)
