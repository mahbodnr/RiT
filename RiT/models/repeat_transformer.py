import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention
from timm.layers import PatchEmbed, Mlp, trunc_normal_
from timm.models.registry import register_model
from .config import DIMS, DEPTHS, HEADS

from .vit import ViTBlock


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).type(
            torch.get_default_dtype()
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class RiT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        depth=3,
        repeats=4,
        heads=6,
        mlp_ratio=4.0,
        pool="cls",
        dropout=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.dim = dim
        self.repeats = repeats
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=dim,
        )
        self.timestep_embedder = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        self.block_output = None

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        x = self.patch_embed(x)  # (B, N, D)
        if self.pool == "cls":
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed

        timesteps = (
            torch.arange(self.repeats, device=x.device, dtype=x.dtype)
            .unsqueeze(1)
            .expand(self.repeats, x.shape[0])
        )  # (T, B)

        self.block_output = []
        for t in timesteps:
            for block in self.blocks:
                x = block(x, self.timestep_embedder(t))
                self.block_output.append(x.clone().detach().cpu())

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        x = self.head(x)
        return x


class SimpleRiT(RiT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        depth=12,
        heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__(
            image_size=img_size,
            patch_size=patch_size,
            channels=channels,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
        )
        del self.timestep_embedder
        self.block = ViTBlock(dim, heads, mlp_ratio=mlp_ratio)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        x = self.patch_embed(x)  # (B, N, D)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed

        for t in range(self.depth):
            x = self.block(x)

        x = self.norm(x)

        if self.cls_token is not None:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.head(x)
        return x


class DiTBlockHalt(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, pool="mean"):
        super().__init__()
        self.pool = pool
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.halt = nn.Linear(hidden_size, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 7 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, halt_mlp = (
            self.adaLN_modulation(c).chunk(7, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if self.pool == "cls":
            halt_prob = torch.sigmoid(self.halt(x[:, 0] * halt_mlp))
        elif self.pool == "mean":
            halt_prob = torch.sigmoid(self.halt(x.mean(dim=1) * halt_mlp))
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        return x, halt_prob

class RiTHalt(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        depth=3,
        max_repeats=10,
        halt_threshold=0.5,
        heads=6,
        mlp_ratio=4.0,
        pool="cls",
        halt_pool="mean",
        dropout=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.dim = dim
        self.depth = depth
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (average pooling)"
        self.pool = pool
        assert max_repeats > 0, "max_repeats must be greater than 0"
        self.max_repeats = max_repeats
        assert (halt_threshold is None) or (0 <= halt_threshold <= 1), "halt_threshold must be in [0, 1]"
        self.halt_threshold = halt_threshold

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=dim,
        )
        self.timestep_embedder = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [DiTBlockHalt(
                dim, 
                heads, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout, 
                pool=halt_pool
            ) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        self.iterations = None

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        x = self.patch_embed(x)  # (B, N, D)
        if self.pool == "cls":
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed

        timesteps = (
            torch.arange(self.max_repeats, device=x.device, dtype=x.dtype)
            .unsqueeze(1)
            .expand(self.max_repeats, x.shape[0])
        )  # (T, B)


        self.iterations = torch.zeros(x.shape[0], device=x.device)
        halt_probs = torch.zeros(x.shape[0], 1, 1, device=x.device)
        self.halt_probs_hist = [] # DELETE LATER
        for t in timesteps:
            for block in self.blocks:
                prev_x = x.clone()

                if self.halt_threshold is None:
                    update_mask = (halt_probs <= torch.rand_like(halt_probs)).squeeze(1).squeeze(1)
                else:
                    update_mask = (halt_probs <= self.halt_threshold).squeeze(1).squeeze(1)
                
                if not update_mask.any():
                    break

                block_halt = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                x[update_mask], block_halt[update_mask] = block(x[update_mask], self.timestep_embedder(t[update_mask]))
                x = x * (1-halt_probs) + prev_x * halt_probs
                halt_probs = halt_probs + (1-halt_probs) * block_halt.unsqueeze(1)
                self.halt_probs_hist.append(halt_probs.clone().detach().cpu()) # DELETE LATER
                self.iterations += update_mask
            if not update_mask.any():
                break

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        x = self.head(x)
        return x


for model in ["tiny", "small", "base", "large", "huge"]:
    for patch_size in [4, 8, 16, 32]:
        for image_size in [32, 64]:
            # RiT
            exec(f"""
@register_model
def rith_d1_{model}_patch{patch_size}_{image_size}(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        image_size={image_size},
        patch_size={patch_size},
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=DIMS["{model}"],
        heads=HEADS["{model}"],
        depth=1,
        halt_threshold=None,
        mlp_ratio=4.0,
        max_repeats=int(DEPTHS["{model}"] * 1.5),
        dropout=kwargs["dropout"],
        halt_pool="cls",
    )
""")
            # RiTHalt
            exec(f"""
@register_model
def rit_d1_{model}_patch{patch_size}_{image_size}(pretrained= False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        image_size={image_size},
        patch_size={patch_size},
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=DIMS["{model}"],
        heads=HEADS["{model}"],
        depth=1,
        mlp_ratio=4.0,
        repeats=DEPTHS["{model}"],
        dropout=kwargs["dropout"],
    )
""")
           