import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from timm.layers import PatchEmbed, Mlp, trunc_normal_

class ViTBlock(nn.Module):
    """
    A Vision Transformer (ViT) block with Multi-Head Self-Attention and MLP.
    """

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=4.0,
        qkv_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = dim

        self.patch_embed = PatchEmbed(
            img_size=image_size,
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
                ViTBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=mlp_dim/dim,
                    qkv_bias=qkv_bias,
                    drop=dpr[i],
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
