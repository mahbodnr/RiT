from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer
from .deepseek import DeepseekMoE
from .configuration_deepseek import DeepseekConfig
import argparse


def getDeepSeekMoEModel(in_features, hidden_features, act_layer, drop):
    config = DeepseekConfig(
        hidden_size=in_features,
        n_shared_experts=2,
        n_routed_experts=16, # 64
        num_experts_per_tok=6, 
        moe_intermediate_size=hidden_features,
    )
    return DeepseekMoE(config=config)


@register_model
def vit_dmoe_tiny_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs,
):
    assert not pretrained, "Pretrained models not available for this model."
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        mlp_layer=getDeepSeekMoEModel,
    )
