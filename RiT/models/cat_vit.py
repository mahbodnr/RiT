import torch
import torch.nn as nn
from timm.layers import PatchEmbed, Mlp, trunc_normal_
from timm.models import register_model



class CatAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            use_v : bool = False,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_v = use_v
        
        if use_v:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if self.use_v:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k = qk.unbind(0)
            v = x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        new_x = attn @ v
        new_x = new_x.transpose(1, 2).reshape(B, N, C)

        new_x = self.proj(new_x)
        new_x = self.proj_drop(new_x)
        x = torch.cat([x, new_x], dim=-1)
        return x
    

class CatViTBlock(nn.Module):
    """
    A Vision Transformer (ViT) block with Multi-Head Self-Attention and MLP.
    """

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, use_v = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # self.attn = CatAttention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     attn_drop=drop,
        #     proj_drop=drop,
        #     use_v = use_v,
        # )
        from timm.models.vision_transformer import Attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(2* dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=2 * dim,
            hidden_features=mlp_hidden_dim,
            out_features=2 * dim,
            act_layer=nn.GELU,
            drop=drop,
        )
        self.proj = nn.Linear(2*dim, dim)

    def forward(self, x):
        x = torch.cat([x ,self.attn(self.norm1(x))], dim = -1)
        x = x + self.mlp(self.norm2(x))
        x = self.proj(x)
        return x


class CatViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        use_v = False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [
            x.item() for x in torch.linspace(0, dropout, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                CatViTBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dpr[i],
                    use_v = use_v,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.head = (
            nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def cat_vit_tiny_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs,
):
    return CatViT(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        heads=3,
        mlp_ratio=4.0,
        **kwargs,
    )

@register_model
def cat_vit_small_patch16_224(
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs,    
    ):
    return CatViT(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )