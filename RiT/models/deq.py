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
import torch.nn.functional as F
from torch.jit import Final

from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.layers import (
    PatchEmbed,
    Mlp,
    LayerType,
    get_act_layer,
    get_norm_layer,
    use_fused_attn,
)

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim should be divisible by num_heads. {dim} % {num_heads} != 0"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionInj(Attention):
    def forward(self, x: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x) + injection.repeat(1, 1, 3)).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
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
        assert drop_path == 0, "drop path not supported in Vision Transformer"
        assert init_values is None, "init values not supported in Vision Transformer"

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError()


class BlockBase(Block):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        return x


class BlockPreNorm(Block):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class BlockPreNormAdd(Block):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + injection
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class BlockAdd(Block):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + injection
        x = x + self.attn(x)
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        return x


class BlockAttn(Block):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = torch.cat([x, injection], dim=1)
        x = x + self.attn(x)
        x = x[:, : -injection.shape[1]]
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        return x


class BlockQKV(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            drop_path,
            init_values,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        self.attn = AttentionInj(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(x, injection)
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        return x

class BlockPreNormQKV(BlockQKV):
    def forward(
        self, x: torch.Tensor, injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm1(x)
        x = x + self.attn(x, injection)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x

class Transit(VisionTransformer):
    """
    Vision Transformer with support for DEQ.
    """

    dynamic_img_size: Final[bool]

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
        # DEQ specific args
        z_init_type: str = "zero",
        f_solver: str = "fixed_point_iter",
        b_solver: str = "fixed_point_iter",
        no_stat: Optional[bool] = None,
        f_max_iter: int = 40,
        b_max_iter: int = 40,
        f_tol: float = 1e-3,
        b_tol: float = 1e-6,
        f_stop_mode: str = "abs",
        b_stop_mode: str = "abs",
        eval_factor: float = 1.0,
        eval_f_max_iter: int = 0,
        ift: bool = False,
        hook_ift: bool = False,
        grad: int = 1,
        tau: float = 1.0,
        sup_gap: int = -1,
        sup_loc: Optional[int] = None,
        n_states: int = 1,
        indexing: Optional[int] = None,
        norm_type: Optional[str] = "weight_norm",
        prefix_filter_out: Optional[Union[str, List[str]]] = None,
        filter_out: Optional[Union[str, List[str]]] = None,
        logger=None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            # DEQ specific args
            f_solver (str, optional): The forward solver function. Default ``'fixed_point_iter'``.
            b_solver (str, optional): The backward solver function. Default  ``'fixed_point_iter'``.
            no_stat (bool, optional): Skips the solver stats computation if True. Default None.
            f_max_iter (int, optional): Maximum number of iterations (NFE) for the forward solver. Default 40.
            b_max_iter (int, optional): Maximum number of iterations (NFE) for the backward solver. Default 40.
            f_tol (float, optional): The forward pass solver stopping criterion. Default 1e-3.
            b_tol (float, optional): The backward pass solver stopping criterion. Default 1e-6.
            f_stop_mode (str, optional): The forward pass fixed-point convergence stop mode. Default ``'abs'``.
            b_stop_mode (str, optional): The backward pass fixed-point convergence stop mode. Default ``'abs'``.
            eval_factor (int, optional): The max iteration for the forward pass at test time, calculated as ``f_max_iter * eval_factor``. Default 1.0.
            eval_f_max_iter (int, optional): The max iteration for the forward pass at test time. Overwrite ``eval_factor`` by an exact number.
            ift (bool, optional): If true, enable Implicit Differentiation. IFT=Implicit Function Theorem. Default False.
            hook_ift (bool, optional): If true, enable a Pytorch backward hook implementation of IFT.
                Furthure reduces memory usage but may affect stability. Default False.
            grad (Union[int, list[int], tuple[int]], optional): Specifies the steps of PhantomGrad.
                It allows for using multiple values to represent different gradient steps in the sampled trajectory states. Default 1.
            tau (float, optional): Damping factor for PhantomGrad. Default 1.0.
            sup_gap (int, optional):
                The gap for uniformly sampling trajectories from PhantomGrad. Sample every ``sup_gap`` states if ``sup_gap > 0``. Default -1.
            sup_loc (list[int], optional):
                Specifies trajectory steps or locations in PhantomGrad from which to sample. Default None.
            n_states (int, optional):
                Uniformly samples trajectory states from the solver.
                The backward passes of sampled states will be automactically tracked.
                IFT will be applied to the best fixed-point estimation when ``ift=True``, while internal states are tracked by PhantomGrad.
                Default 1. By default, only the best fixed point estimation will be returned.
            indexing (int, optional):
                Samples specific trajectory states at the given steps in ``indexing`` from the solver. Similar to ``n_states`` but more flexible.
                Default None.
            norm_type (str, optional): Type of normalization to be applied. Default is ``'weight_norm'``.
            prefix_filter_out (list or str, optional):
                List of module weights prefixes to skip out when applying normalization. Default is None.
            filter_out (list or str, optional):
                List of module weights names to skip out when applying normalization. Default is None.
        """
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
        assert drop_path_rate == 0, "drop path not supported in Vision Transformer"
        assert init_values is None, "init values not supported in Vision Transformer"
        assert z_init_type in ("zero", "input"), "Invalid z_init_type"
        self.logger = logger
        self.z_init_type = z_init_type
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

        self.injection_transform = nn.Linear(embed_dim, embed_dim)
        self.deq = get_deq(
            f_solver=f_solver,
            b_solver=b_solver,
            no_stat=no_stat,
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_stop_mode=f_stop_mode,
            b_stop_mode=b_stop_mode,
            eval_factor=eval_factor,
            eval_f_max_iter=eval_f_max_iter,
            ift=ift,
            hook_ift=hook_ift,
            grad=grad,
            tau=tau,
            sup_gap=sup_gap,
            sup_loc=sup_loc,
            n_states=n_states,
            # indexing=indexing,
        )

        apply_norm(
            self.blocks,
            norm_type=norm_type,
            prefix_filter_out=prefix_filter_out,
            filter_out=filter_out,
        )

    def deq_forward_writer(self, info):
        if self.logger is not None:
            for key in ("abs_lowest", "rel_lowest", "nstep"):
                if key == "nstep" and info[key].mean() == 0:
                    continue
                if info[key].mean() == 1e8:
                    continue
                self.logger(f"Forward/{key}", info[key].mean())

    def deq_backward_writer(self, info):
        if self.logger is not None:
            for key in ("abs_lowest", "rel_lowest", "nstep"):
                self.logger(f"Backward/{key}", info[key].mean())

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        raise NotImplementedError("grad_checkpointing is not available for DEQ models")

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n
        )
        raise NotImplementedError("Not Implemented Yet for DEQ")

    def _init_z(self, x: torch.Tensor) -> torch.Tensor:
        if self.z_init_type == "zero":
            return torch.zeros_like(x)
        elif self.z_init_type == "input":
            return x

    def injection(self, x: torch.Tensor) -> torch.Tensor:
        return self.injection_transform(x)

    def run_deq(self, x: torch.Tensor) -> torch.Tensor:
        reset_norm(self.blocks)
        u_inj = self.injection(x)

        def deq_func(z):
            for block in self.blocks:
                z = block(z, injection=u_inj)
            return z

        z_init = self._init_z(x)
        z, info = self.deq(deq_func, z_init, writer=self.deq_backward_writer)
        self.deq_forward_writer(info)
        assert len(z) == 1  # DELETE LATER
        return z[-1]

        # z = self._init_z(x)
        # for t in range(self.deq.f_max_iter):
        #     for block in self.blocks:
        #         z = block(z, injection=u_inj)
        # return z

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.run_deq(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


_blocks = {
    "base": BlockBase,
    "prenorm": BlockPreNorm,
    "prenorm_add": BlockPreNormAdd,
    "add": BlockAdd,
    "attn": BlockAttn,
    "qkv": BlockQKV,
    "prenorm_qkv": BlockPreNormQKV,
}


@register_model
def transit_tiny_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> Transit:
    assert not pretrained, "Pretrained model not available for this configuration"
    return Transit(
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
def transit_small_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> Transit:
    assert not pretrained, "Pretrained model not available for this configuration"
    return Transit(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        block_fn=_blocks[kwargs["block_type"]],
        **kwargs,
    )


@register_model
def transit_base_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs: Any,
) -> Transit:
    assert not pretrained, "Pretrained model not available for this configuration"
    return Transit(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        block_fn=_blocks[kwargs["block_type"]],
        **kwargs,
    )
