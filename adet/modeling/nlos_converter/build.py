# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

NLOS_CONVERTER_REGISTRY = Registry("NLOS_CONVERTER")
NLOS_CONVERTER_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

#from . import conv_fc_nlos_converter  # noqa F401 isort:skip


def build_nlos_converter(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.NLOS_CONVERTER.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.NLOS_CONVERTER.NAME
    return NLOS_CONVERTER_REGISTRY.get(name)(cfg, input_shape)
