from timm.models.registry import register_model
# from .vit_pytorch.vit import ViT
from .vit import ViT
    
@register_model
def vit_tiny_patch4_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch4_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch8_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch8_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch16_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch16_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch32_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_tiny_patch32_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0,
    )

@register_model
def vit_small_patch4_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch4_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch8_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch8_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch16_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch16_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch32_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_small_patch32_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0,
    )

@register_model
def vit_base_patch4_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch4_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch8_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch8_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch16_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch16_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch32_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_base_patch32_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0,
    )

@register_model
def vit_large_patch4_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch4_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch8_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch8_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch16_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch16_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch32_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_large_patch32_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0,
    )

@register_model
def vit_huge_patch4_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch4_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch8_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch8_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch16_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch16_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch32_32(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )

@register_model
def vit_huge_patch32_64(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        depth=32,
        heads=24,
        mlp_dim=6144,
        dropout=0,
    )