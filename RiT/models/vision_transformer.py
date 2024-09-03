from timm.models.registry import register_model
from .vit_pytorch.vit import ViT
from .config import DIMS, DEPTHS, HEADS, MLP_DIMS

for model in ["tiny", "small", "base", "large", "huge"]:
    for patch_size in [4, 8, 16, 32]:
        for image_size in [32, 64]:
                exec(f"""
@register_model
def vit_{model}_patch{patch_size}_{image_size}(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=DIMS[model],
        depth=DEPTHS[model],
        heads=HEADS[model],
        mlp_dim=MLP_DIMS[model],
        dropout=kwargs["dropout"],
    )
""")