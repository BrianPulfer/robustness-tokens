from models.dinov2 import SUPPORTED_DINOV2_MODELS, get_model as get_dinov2_model
from models.openclip import SUPPORTED_OPENCLIP_MODELS, get_model as get_openclip_model
from models.deit3 import SUPPORTED_DEIT3_MODELS, get_model as get_deit3_model

ALL_SUPPORTED_MODELS = SUPPORTED_DINOV2_MODELS + SUPPORTED_OPENCLIP_MODELS + SUPPORTED_DEIT3_MODELS


def get_model(**kwargs):
    model_name = kwargs["name"]

    if model_name not in ALL_SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {ALL_SUPPORTED_MODELS}"
        )

    if model_name in SUPPORTED_DINOV2_MODELS:
        return get_dinov2_model(**kwargs)
    
    if model_name in SUPPORTED_OPENCLIP_MODELS:
        return get_openclip_model(**kwargs)
    
    if model_name in SUPPORTED_DEIT3_MODELS:
        return get_deit3_model(**kwargs)
