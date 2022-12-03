from .RestNet34 import *
from .Swin_T import *
from .ViTAE_S import *


def build_model(model_arch, **kwargs):
    if model_arch == "r34":
        model = p3mnet_resnet34(**kwargs)
    elif model_arch == "swin":
        model = p3mnet_swin_t(**kwargs)
    elif model_arch == "vitae":
        model = p3mnet_vitae_s(**kwargs)
    else:
        raise NotImplementedError

    return model
