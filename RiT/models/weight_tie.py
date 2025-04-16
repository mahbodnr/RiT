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

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, VisionTransformer
from timm.layers import Mlp, PatchEmbed, get_act_layer, get_norm_layer, LayerType
from timm.models import register_model


def get_nested_attribute(module, name):
    attributes = name.split(".")
    for attr in attributes:
        module = getattr(module, attr)
    return module


class WeightTieBlock(nn.Module):
    """A Transformer block with weight tying between the attention and MLP layers."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        distil_mode: Literal["avg",] = "avg",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[Callable] = nn.GELU,
        norm_layer: Type[Callable] = nn.LayerNorm,
        mlp_layer: Type[Callable] = Mlp,
    ) -> None:
        super().__init__()

        self.weight_tie = False
        self.depth = depth
        self.distil_mode = distil_mode

        self.relaxed_blocks = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(depth)
            ]
        )

        self.weight_tie_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.weight_tie:
            for _ in range(self.depth):
                x = self.weight_tie_block(x, **kwargs)
        else:
            x = self.relaxed_blocks(x, **kwargs)
        return x


    def tie_weights(self) -> None:
        self.weight_tie = True

        if self.distil_mode == "avg":
            for name, p in self.weight_tie_block.named_parameters():
                blocks = torch.stack(
                    [get_nested_attribute(block, name) for block in self.relaxed_blocks]
                )
                p.data = torch.mean(blocks, dim=0).data


    def relax_weights(self) -> None:
        self.weight_tie = False

        for name, p in self.weight_tie_block.named_parameters():
            for block in self.relaxed_blocks:
                get_nested_attribute(block, name).data = p.data


class WTViT(VisionTransformer):
    def __init__(
            self,
            distil_mode: str = 'avg',
            n_pre_layers: int = 0,
            n_post_layers: int = 0,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            mlp_layer: Type[nn.Module] = Mlp,
            block_fn: Type[Block] = Block,
            **kwargs: Any,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            final_norm=final_norm,
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
            mlp_layer=mlp_layer,
            block_fn=block_fn,
        )
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.blocks = WeightTieBlock(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            distil_mode=distil_mode,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            init_values=init_values,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.pre_blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(n_pre_layers)
            ]
        )

        self.post_blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(n_post_layers)
            ]
        )


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.pre_blocks(x)
        x = self.blocks(x)
        x = self.post_blocks(x)
        x = self.norm(x)
        return x
    
    def tie_weights(self) -> None:
        self.blocks.tie_weights()
        
    def relax_weights(self) -> None:
        self.blocks.relax_weights()

    @property
    def weight_tie(self) -> bool:
        return self.blocks.weight_tie
        

@register_model
def wtvit_tiny_patch16_224(
    pretrained: bool = False,
    **kwargs: Any,
) -> WTViT:
    """ WTViT-Tiny model from `"Weight-Tied Vision Transformer"`"""
    model = WTViT(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        **kwargs
    )
    return model
