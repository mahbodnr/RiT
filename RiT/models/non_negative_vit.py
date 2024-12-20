import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from timm.models import register_model

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def positive_norm(t, dim = -1):
    return F.normalize(F.relu(t), dim = dim, p = 1)

# for use with parametrize

class PositiveNorm(Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return positive_norm(t, dim = self.dim)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True,
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            PositiveNorm(dim = -1 if norm_dim_in else 0)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# attention and feedforward
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert not qk_norm 
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.weight_q = NormLinear(dim, dim, bias=qkv_bias)
        self.weight_k = NormLinear(dim, dim, bias=qkv_bias)
        self.weight_v = NormLinear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.weight_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.weight_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.weight_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.unbind(0)
        print(self.weight_q(x).sum(-1))
        exit()

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=False,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        assert not bias
        assert norm_layer is None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = NormLinear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = NormLinear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class nViTBlock(Module):
    def __init__(
        self,
        embed_dim,
        dim_head,
        num_heads,
        mlp_ratio = 4,
        dropout = 0.,
        residual_lerp_scale_init = 1, # 1 / depth
    ):
        super().__init__()

        self.dim = embed_dim
        self.scale = embed_dim ** 0.5
        mlp_dim = int(embed_dim * mlp_ratio)

        self.attn = Attention(embed_dim, dim_head = dim_head, heads = num_heads, dropout = dropout)
        self.ff = FeedForward(embed_dim, dim_inner = mlp_dim, dropout = dropout)



    def forward(self, x):
        x = positive_norm(x)
        x = (x + self.attn(x)) * 0.5
        x = (x + self.ff(x)) * 0.5

        return x

class nViT(Module):
    """ https://arxiv.org/abs/2410.01131 """

    def __init__(
        self,
        *,
        img_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout = 0.,
        channels = 3,
        dim_head = 64,
        residual_lerp_scale_init = None,
        **kwargs
    ):
        super().__init__()
        image_height, image_width = pair(img_size)

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)
        num_patches = patch_height_dim * patch_width_dim
        mlp_dim = int(embed_dim * mlp_ratio)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            NormLinear(patch_dim, embed_dim, norm_dim_in = False),
        )

        self.abs_pos_emb = NormLinear(embed_dim, num_patches)

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        # layers

        self.dim = embed_dim
        self.scale = embed_dim ** 0.5

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(embed_dim, dim_head = dim_head, heads = num_heads, dropout = dropout),
                FeedForward(embed_dim, dim_inner = mlp_dim, dropout = dropout),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale),
            ]))

        self.logit_scale = nn.Parameter(torch.ones(num_classes))

        self.to_pred = NormLinear(embed_dim, num_classes)

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            normed = module.weight
            original = module.linear.parametrizations.weight.original

            original.copy_(normed)

    def forward(self, images):
        device = images.device

        tokens = self.to_patch_embedding(images)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device = device)]

        tokens = l2norm(tokens + pos_emb)

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):

            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        pooled = reduce(tokens, 'b n d -> b d', 'mean')

        logits = self.to_pred(pooled)
        logits = logits * self.logit_scale * self.scale

        return logits

# @register_model
# def nvit_tiny_patch16_224(
#     num_classes: int = 1000,
#     pretrained: bool = False,
#     **kwargs,
# ) -> nViT:
#     model = nViT(
#         img_size=224,
#         patch_size=16,
#         embed_dim=192,
#         depth=12,
#         num_heads=3,
#         mlp_ratio=4,
#         num_classes=num_classes,
#         **kwargs
#     )
#     return model