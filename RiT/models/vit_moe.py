from st_moe_pytorch import MoE

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
from timm.models.vision_transformer import VisionTransformer, LayerScale, DropPath, Attention
from timm.layers import Mlp, LayerType, PatchEmbed
from timm.models import register_model

class BlockMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = MoE,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            dim=dim,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.total_aux_loss = None

    def forward(self, x: torch.Tensor, injection=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = self.norm2(x)
        x, self.total_aux_loss, balance_loss, router_z_loss = self.mlp(x)
        x = x + self.drop_path2(self.ls2(x))
        return x


class ViTMoE(VisionTransformer):
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
        # MOE specific args
        num_experts: int = 10,
        gating_top_n: int = 2,
        threshold_train: float = 0.2,
        threshold_eval: float = 0.2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        balance_loss_coef: float = 1e-2,
        router_z_loss_coef: float = 1e-3,
        **kwargs: Any,
    ):

        moe = partial(
            MoE, 
            num_experts=num_experts,
            expert_hidden_mult=mlp_ratio,
            gating_top_n=gating_top_n,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            balance_loss_coef=balance_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
        )
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
            block_fn=BlockMoE,
            mlp_layer=moe,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        self.aux_loss = sum([block.total_aux_loss for block in self.blocks]) / len(self.blocks)
        return x


@register_model
def vit_moe_tiny_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> ViTMoE:
    model = ViTMoE(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs
    )
    return model