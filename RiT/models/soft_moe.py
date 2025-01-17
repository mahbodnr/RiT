from soft_mixture_of_experts.vit import _build_soft_moe_vit
from timm.models import register_model

@register_model
def vit_soft_moe_tiny_patch16_224(pretrained= False, **kwargs):
    model = _build_soft_moe_vit(
        num_classes= 1000,
        image_size= 224,
        patch_size= 16,
        d_model= 192,
        nhead= 3,
        num_encoder_layers= 12,
        num_experts= 64,
    )
    return model

@register_model
def vit_soft_moe_small_patch16_224(pretrained= False, **kwargs):
    model = _build_soft_moe_vit(
        num_classes= 1000,
        image_size= 224,
        patch_size= 16,
        d_model= 384,
        nhead= 6,
        num_encoder_layers= 12,
        num_experts= 128,
    )
    return model