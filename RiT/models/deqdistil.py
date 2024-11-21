from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.models.deit import VisionTransformerDistilled
from timm.models import register_model

from .deq import Transit, _blocks


class TransitDistilled(Transit, VisionTransformerDistilled):
    """Transit distilled model."""

@register_model
def transit_tiny_distil_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> TransitDistilled:
    assert not pretrained, "Pretrained model not available for this configuration"
    return TransitDistilled(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        num_classes=num_classes,
        block_fn=_blocks[kwargs["block_type"]],
        **kwargs,
    )

@register_model
def transit_small_distil_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> TransitDistilled:
    assert not pretrained, "Pretrained model not available for this configuration"
    return TransitDistilled(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        block_fn=_blocks[kwargs["block_type"]],
        **kwargs,
    )