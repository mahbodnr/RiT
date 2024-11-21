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

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim = self.dim)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim = -1 if norm_dim_in else 0)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# attention and feedforward

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)

        self.dropout = dropout

        self.q_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))
        self.k_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False)

    def forward(
        self,
        x
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # query key rmsnorm

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        # scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25) - eq. 16

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.,
            scale = 1.
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        dim_inner,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim_inner * 2 / 3)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale
        gate = gate * self.gate_scale * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden

        hidden = self.dropout(hidden)
        return self.to_out(hidden)


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

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        self.attn = Attention(embed_dim, dim_head = dim_head, heads = num_heads, dropout = dropout)
        self.ff = FeedForward(embed_dim, dim_inner = mlp_dim, dropout = dropout)

        self.attn_alpha = nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale)
        self.ff_alpha = nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale)


    def forward(self, x):
        attn_out = l2norm(self.attn(x))
        x = l2norm(x.lerp(attn_out, self.attn_alpha * self.scale))

        ff_out = l2norm(self.ff(x))
        x = l2norm(x.lerp(ff_out, self.ff_alpha * self.scale))

        return x


class nViTBlock2(Module):
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

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        self.attn = Attention(embed_dim, dim_head = dim_head, heads = num_heads, dropout = dropout)
        self.ff = FeedForward(embed_dim, dim_inner = mlp_dim, dropout = dropout)

        self.attn_alpha = nn.Linear(embed_dim, embed_dim)
        self.ff_alpha = nn.Linear(embed_dim, embed_dim)

        self.gelu = nn.GELU()


    def forward(self, x):
        attn_alpha = self.gelu(self.attn_alpha(x)).mean(dim = -2, keepdim = True)
        attn_out = l2norm(self.attn(x))
        x = l2norm(x.lerp(attn_out, attn_alpha * self.scale))

        ff_alpha = self.gelu(self.ff_alpha(x)).mean(dim = -2, keepdim = True)
        ff_out = l2norm(self.ff(x))
        x = l2norm(x.lerp(ff_out, ff_alpha * self.scale))

        return x


# classes

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