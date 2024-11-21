from RiT.models.normalized_vit import nViTBlock, L2Norm, NormLinear

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    List,
    Literal,
)
from functools import partial
from typing import Callable, Optional, Tuple, Union, Type

import torch
from torch import nn
from torch.jit import Final
from timm.models.vision_transformer import (
    VisionTransformer,
    LayerScale,
    DropPath,
    Attention,
    Block,
)
from timm.layers import Mlp, LayerType, PatchEmbed
from timm.models import register_model


class nViT(VisionTransformer):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 1,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        # nViT args
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )

        self.scale = embed_dim ** 0.5
        self.pre_norm = L2Norm(embed_dim)
        self.blocks = nn.Sequential(
            *[
                nViTBlock(
                    embed_dim=embed_dim,
                    dim_head=embed_dim // num_heads,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=drop_rate,
                    residual_lerp_scale_init=1/depth,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.Identity()
        self.head = NormLinear(embed_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.ones(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.logit_scale * self.scale

@register_model
def nvit_tiny_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> nViT:
    model = nViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs,
    )
    return model
