import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention
from timm.layers import PatchEmbed, Mlp, trunc_normal_
from timm.models.registry import register_model

from timm.models.vision_transformer import Block as ViTBlock

from torchdeq.norm import apply_norm, reset_norm

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

    @torch.compile
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
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
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
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

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class RiT(nn.Module):
    def __init__(
        self,
        img_size=224,
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

        self.timesteps = torch.nn.parameter.Parameter(
            torch.arange(self.repeats).unsqueeze(1),
            requires_grad=False,
        )  # (T, B)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
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
            [
                DiTBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for i in range(depth)
            ]
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

        timesteps = self.timesteps.expand(self.repeats, x.shape[0])  # (T, B)
        # self.block_output = []
        for t in timesteps:
            for block in self.blocks:
                x = block(x, self.timestep_embedder(t))
                # self.block_output.append(x.clone().detach().cpu())

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        x = self.head(x)
        return x


class SimpleRiT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        repeats=12,
        depth=1,
        heads=6,
        mlp_ratio=4.0,
        pool="cls",
        halt=None,
        ema_alpha=0.5,
        halt_threshold=0.9,
        halt_noise_scale=1,
        dropout=0.0,
        stochastic_depth=False,
        extra_step=False, # DELETE LATER
        normalize=False, # DELETE LATER
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.dim = dim
        self.repeats = repeats
        self.extra_step = extra_step # DELETE LATER
        self.normalize = normalize # DELETE LATER
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.stochastic_depth = stochastic_depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=dim,
        )

        self.num_patches = (
            self.patch_embed.num_patches + (pool == "cls") + (halt is not None)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        trunc_normal_(self.pos_embed, std=0.02)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim,
                    heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        # halt mechanism
        assert halt in {
            None,
            "ema",
            "add",
            "classify",
        }, f"halt type must be either None, ema, add, or classify. Got {halt}"
        self.halt = halt
        if halt is not None:
            assert 0 <= halt_threshold <= 1, "halt_threshold must be in [0, 1]"
            self.halt_threshold = halt_threshold
            self.halt_token = nn.Parameter(torch.zeros(1, 1, dim))
            trunc_normal_(self.halt_token, std=0.02)
            self.halt_block = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.SiLU(),
                nn.Linear(dim, 1, bias=False),
                # nn.Sigmoid(),
            )
            self.halt_noise_scale = halt_noise_scale

        if halt == "ema":
            self.ema_alpha = ema_alpha

    def embed_input(self, x):
        x = self.patch_embed(x)  # (B, N, D)
        if self.cls_token is not None:
            x = torch.cat(
                (self.cls_token.expand(x.shape[0], -1, -1), x), dim=1
            )  # put cls token at the beginning
        if self.halt is not None:
            x = torch.cat(
                (x, self.halt_token.expand(x.shape[0], -1, -1)), dim=1
            )  # put halt token at the end
        x = x + self.pos_embed

        return x

    def classify(self, x):
        x = self.norm(x)
        if self.cls_token is not None:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]
        x = self.head(x)

        return x

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        x = self.embed_input(x)
        input = x.clone()

        if self.halt is not None:
            if self.halt == "classify":
                output = torch.zeros(
                    x.shape[0], self.num_classes, device=x.device, dtype=x.dtype
                )
            else:
                halt_probs = x.new_zeros((x.shape[0], 1, 1))
                halted = torch.zeros_like(halt_probs, dtype=torch.bool)

        if self.stochastic_depth:
            repeats = (
                torch.round(torch.normal(self.repeats, self.repeats / 5, (1,)))
                .int()
                .clamp(min=1)
                .item()
            )
        else:
            repeats = self.repeats
        for t in range(repeats):
            for block in self.blocks:
                prev_x = x.clone()
                x = block(x + input)
                if self.normalize:
                    x = (x - x.mean(dim=(-1, -2), keepdim=True)) / x.std(
                        dim=(-1, -2), keepdim=True
                    )

                if self.halt is not None:
                    if self.halt == "classify":
                        block_halt = self.halt_block(x[:, -1])
                        block_halt_prob = torch.sigmoid(
                            block_halt
                            + self.halt_noise_scale * torch.randn_like(block_halt)
                        )
                        output = output + block_halt_prob * self.classify(x)
                    else:
                        x = torch.lerp(x, prev_x, halt_probs)
                        block_halt = self.halt_block(x[:, -1])
                        block_halt_prob = torch.sigmoid(
                            block_halt
                            + self.halt_noise_scale * torch.randn_like(block_halt)
                        ).unsqueeze(1)

                        if self.halt == "ema":
                            halt_probs = torch.lerp(
                                halt_probs, block_halt_prob, self.ema_alpha
                            )
                        elif self.halt == "add":
                            halt_probs = halt_probs + (1 - halt_probs) * block_halt_prob

                        if self.halt_threshold < 1:
                            halted = halted + (halt_probs >= self.halt_threshold)
                            halt_probs = halted + halt_probs * ~halted

        if self.halt == "classify":
            return output

        output = self.classify(x)
        return output

    def inference(
        self, x, repeats, halt, halt_threshold, ema_alpha, halt_noise_scale=0
    ):
        halt_probs_hist = []
        halted_hist = []
        block_halt_hist = []
        x = self.embed_input(x)
        input = x.clone()

        if halt is not None:
            if halt == "classify":
                output = torch.zeros(
                    x.shape[0], self.num_classes, device=x.device, dtype=x.dtype
                )
            else:
                halt_probs = x.new_zeros((x.shape[0], 1, 1))
                halted = torch.zeros_like(halt_probs, dtype=torch.bool)

        layer_outputs = []
        block_outputs = []
        for t in range(repeats):
            for block in self.blocks:
                prev_x = x.clone()
                x = block(x + input)
                if self.normalize:
                    x = (x - x.mean(dim=(-1, -2), keepdim=True)) / x.std(
                        dim=(-1, -2), keepdim=True
                    )
                block_outputs.append(x.clone().detach())

                if halt is not None:
                    if halt == "classify":
                        block_halt = self.halt_block(x[:, -1])
                        block_halt_prob = torch.sigmoid(
                            block_halt + halt_noise_scale * torch.randn_like(block_halt)
                        )
                        layer_outputs.append(self.classify(x))
                        output = output + block_halt_prob * self.classify(x)
                    else:
                        x = torch.lerp(x, prev_x, halt_probs)
                        block_halt = self.halt_block(x[:, -1])
                        block_halt_prob = torch.sigmoid(
                            block_halt + halt_noise_scale * torch.randn_like(block_halt)
                        ).unsqueeze(1)

                        if halt == "ema":
                            halt_probs = torch.lerp(
                                halt_probs, block_halt_prob, ema_alpha
                            )
                        elif halt == "add":
                            halt_probs = halt_probs + (1 - halt_probs) * block_halt_prob

                        if halt_threshold < 1:
                            halted = halted + (halt_probs >= halt_threshold)
                            halt_probs = halted + halt_probs * ~halted

                        halt_probs_hist.append(halt_probs.clone().detach())
                        halted_hist.append(halted.clone().detach())
                    block_halt_hist.append(block_halt_prob.clone().detach())

        if halt == "classify":
            return {
                "logits": output,
                "layer_outputs": layer_outputs,
                "halt_probs": halt_probs_hist,
                "halted": halted_hist,
                "block_halt": block_halt_hist,
                "block_outputs": block_outputs,
            }
        x = self.classify(x)
        return {
            "logits": x,
            "halt_probs": halt_probs_hist,
            "halted": halted_hist,
            "block_halt": block_halt_hist,
            "block_outputs": block_outputs,
        }


class SimpleRiT2(SimpleRiT):
    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        assert self.halt == "classify", "SimpleRiT2 only supports halt='classify'"
        x = self.embed_input(x)

        outputs = []
        block_outputs = [] # DELETE LATER
        confidences = []
        if self.stochastic_depth:
            repeats = (
                torch.round(torch.normal(self.repeats, self.repeats / 5, (1,)))
                .int()
                .clamp(min=1)
                .item()
            )
        else:
            repeats = self.repeats
        for t in range(repeats):
            for block in self.blocks:
                x = block(x)
                if self.normalize:
                    x = (x - x.mean(dim=(-1, -2), keepdim=True)) / x.std(
                        dim=(-1, -2), keepdim=True)
                # block_halt = self.halt_block(x[:, -1])
                # block_halt = torch.sigmoid(block_halt + self.halt_noise_scale * torch.randn_like(block_halt))
                block_outputs.append(x.clone().detach()) # DELETE LATER
                outputs.append(self.classify(x))
                # confidences.append(block_halt.squeeze(-1))

        if self.extra_step:
            x = block(x)
            if self.normalize:
                x = (x - x.mean(dim=(-1, -2), keepdim=True)) / x.std(
                    dim=(-1, -2), keepdim=True)
            outputs.append(self.classify(x))

        block_outputs = torch.stack(block_outputs) # DELETE LATER
        outputs = torch.stack(outputs)
        # confidences = torch.stack(confidences)
        # confidences = F.softmax(confidences, dim=0)
        # confidences = torch.cumsum(confidences, dim=0)

        return {
            "logits": outputs,  # (N, B, C)
            # "confidences": confidences,  # (N, B)
            "block_outputs": block_outputs,  # DELETE LATER
        }

from timm.models.vision_transformer import Attention,Mlp
class ViTBlockIm(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: float = None,
            drop_path: float = 0.,
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

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor, inj=None) -> torch.Tensor:
        x = x + inj
        x = x + self.attn(x)
        x = x + self.mlp(self.norm1(x))
        x = self.norm2(x)
        return x


class SimpleRiT3(SimpleRiT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        repeats=12,
        depth=1,
        heads=6,
        mlp_ratio=4.0,
        pool="cls",
        halt=None,
        ema_alpha=0.5,
        halt_threshold=0.9,
        halt_noise_scale=1,
        dropout=0.0,
        stochastic_depth=False,
        extra_step=False, # DELETE LATER
        normalize=False, # DELETE LATER
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            channels=channels,
            num_classes=num_classes,
            dim=dim,
            repeats=repeats,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            pool=pool,
            halt=halt,
            ema_alpha=ema_alpha,
            halt_threshold=halt_threshold,
            halt_noise_scale=halt_noise_scale,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            extra_step=extra_step, # DELETE LATER
            normalize=normalize, # DELETE LATER
        )

        self.injection = nn.Linear(dim, dim, bias=True)
        self.blocks = nn.ModuleList(
            [
                ViTBlockIm(
                    dim,
                    heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )
        if normalize:
            apply_norm(
                self.blocks, 
                # "weight_norm",
                norm_type = "none",
            )



    def forward(self, x):
        """
        Forward pass of the model.

        :param x: a (B, H, W, C) image tensor.
        :return: a (B, num_classes) tensor of logits.
        """
        assert self.halt == None, "SimpleRiT3 does not support halt"
        
        x = self.embed_input(x)
        injection = self.injection(x)

        outputs = []
        block_outputs = [] # DELETE LATER
        z = torch.zeros_like(x)
        reset_norm(self.blocks)

        if self.stochastic_depth:
            repeats = (
                torch.round(torch.normal(self.repeats, self.repeats / 5, (1,)))
                .int()
                .clamp(min=1)
                .item()
            )
        else:
            repeats = self.repeats
        for t in range(repeats):
            for block in self.blocks:
                z = block(z, injection)
                block_outputs.append(z.clone().detach()) # DELETE LATER
                outputs.append(self.classify(z))

        block_outputs = torch.stack(block_outputs) # DELETE LATER
        outputs = torch.stack(outputs)

        return {
            "logits": outputs,  # (N, B, C)
            "block_outputs": block_outputs,  # DELETE LATER
        }

class DiTBlockHalt(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, pool="mean"):
        super().__init__()
        assert pool in {
            None,
            "mean",
            "cls",
        }, "halt_pool type must be either cls (cls token) or mean (average pooling)"
        self.pool = pool
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
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
        # self.halt = nn.Linear(hidden_size, 1, bias=True)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.halt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size, bias=True),
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_halt,
            scale_halt,
        ) = self.adaLN_modulation(c).chunk(8, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if self.pool == "cls":
            halt_x = x[:, 0:1]
        elif self.pool == "mean":
            halt_x = x.mean(dim=1, keepdim=True)
        else:
            halt_x = x

        halt_prob = self.halt(self.modulate(self.norm3(halt_x), shift_halt, scale_halt))
        return x, halt_prob


class RiTHalt(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=384,
        depth=3,
        max_repeats=10,
        halt_threshold=0.95,
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
            None,
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (average pooling)"
        self.pool = pool
        self.halt_pool = halt_pool
        self.max_repeats = max_repeats
        assert (halt_threshold is None) or (
            0 <= halt_threshold <= 1
        ), "halt_threshold must be in [0, 1]"
        self.halt_threshold = halt_threshold

        self.timesteps = torch.nn.parameter.Parameter(
            torch.arange(self.max_repeats).unsqueeze(1),
            requires_grad=False,
        )  # (T, B)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
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
            [
                DiTBlockHalt(
                    dim, heads, mlp_ratio=mlp_ratio, dropout=dropout, pool=halt_pool
                )
                for _ in range(depth)
            ]
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

        if self.halt_pool is None:
            halt_probs = x.new_zeros((x.shape[0], x.shape[1], 1))
        else:
            halt_probs = x.new_zeros((x.shape[0], 1, 1))
        halted = torch.zeros_like(halt_probs, dtype=torch.bool)
        timesteps = self.timesteps.expand(self.max_repeats, x.shape[0])  # (T, B)
        for t in timesteps:
            for block in self.blocks:
                prev_x = x.clone()
                x, block_halt = block(x, self.timestep_embedder(t))
                x = torch.lerp(
                    x, prev_x, halt_probs
                )  # = x * (1-halt_probs) + prev_x * halt_probs
                # add-up approach:
                # halt_probs = halt_probs + (1-halt_probs) * block_halt
                # EMA approach:
                alpha = 0.5
                halt_probs = torch.lerp(halt_probs, block_halt, alpha)
                halted = halted + (halt_probs > self.halt_threshold)
                halt_probs = halted + halt_probs * ~halted

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        x = self.head(x)
        return x

    # def forward(self, x):
    #     """
    #     Forward pass of the model.

    #     :param x: a (B, H, W, C) image tensor.
    #     :return: a (B, num_classes) tensor of logits.
    #     """
    #     x = self.patch_embed(x)  # (B, N, D)
    #     if self.pool == "cls":
    #         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     x = x + self.pos_embed

    #     if self.halt_pool is None:
    #         halt_probs = x.new_zeros((x.shape[0], x.shape[1], 1))
    #     else:
    #         halt_probs = x.new_zeros((x.shape[0], 1, 1))
    #     halted = torch.zeros_like(halt_probs, dtype=torch.bool)
    #     timesteps = self.timesteps.expand(self.max_repeats, x.shape[0])  # (T, B)
    #     for t in timesteps:
    #         for block in self.blocks:
    #             prev_x = x

    #             # update x
    #             still_running = ~halted.view(-1)
    #             block_halt = torch.zeros_like(halt_probs)
    #             x[still_running], block_halt[still_running] = block(x[still_running], self.timestep_embedder(t)[still_running])
    #             x[still_running] = torch.lerp(x[still_running], prev_x[still_running], halt_probs[still_running]) # = x * (1-halt_probs) + prev_x * halt_probs
    #             # update halt_probs:
    #             # add-up approach:
    #             halt_probs = halt_probs + (1-halt_probs) * block_halt
    #             # EMA approach:
    #             # alpha = 0.5
    #             # halt_probs = torch.lerp(halt_probs, block_halt, alpha)

    #             halted = halted + (halt_probs > self.halt_threshold)
    #             halt_probs = halted + halt_probs * ~halted

    #     x = self.norm(x)

    #     if self.pool == "mean":
    #         x = x.mean(dim=1)
    #     elif self.pool == "cls":
    #         x = x[:, 0]
    #     else:
    #         raise NotImplementedError(f"pool type {self.pool} not implemented")

    #     x = self.head(x)
    #     return x

    def inference(self, x, max_repeats, halt_threshold, alpha):
        x = self.patch_embed(x)  # (B, N, D)
        if self.pool == "cls":
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed

        if self.halt_pool is None:
            halt_probs = x.new_zeros((x.shape[0], x.shape[1], 1))
        else:
            halt_probs = x.new_zeros((x.shape[0], 1, 1))
        halted = torch.zeros_like(halt_probs, dtype=torch.bool)
        halt_probs_hist = []
        halted_hist = []
        block_halt_hist = []
        timesteps = (
            torch.arange(
                max_repeats,
                device=x.device,
            )
            .unsqueeze(1)
            .expand(max_repeats, x.shape[0])
        )  # (T, B)
        for t in timesteps:
            for block in self.blocks:
                prev_x = x.clone()

                # update x
                x, block_halt = block(x, self.timestep_embedder(t))
                x = torch.lerp(
                    x, prev_x, halt_probs
                )  # = x * (1-halt_probs) + prev_x * halt_probs
                # update halt_probs:
                # add-up approach:
                # halt_probs = halt_probs + (1-halt_probs) * block_halt
                # EMA approach:
                halt_probs = torch.lerp(halt_probs, block_halt, alpha)
                halted = halted + (halt_probs > halt_threshold)
                halt_probs = halted + halt_probs * ~halted

                # monitor halting probabilities
                halt_probs_hist.append(halt_probs.clone().detach())
                halted_hist.append(halted.clone().detach())
                block_halt_hist.append(block_halt.clone().detach())

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        else:
            raise NotImplementedError(f"pool type {self.pool} not implemented")

        x = self.head(x)
        return {
            "logits": x,
            "halt_probs": halt_probs_hist,
            "halted": halted_hist,
            "block_halt": block_halt_hist,
        }


@register_model
def rit_d1_tiny_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_tiny_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_small_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_base_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        mlp_ratio=4.0,
        repeats=12,
        dropout=0,
    )


@register_model
def rit_d1_large_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_large_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        mlp_ratio=4.0,
        repeats=24,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rit_d1_huge_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiT(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        mlp_ratio=4.0,
        repeats=32,
        dropout=0,
    )


@register_model
def rith_d1_tiny_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool=None,
    )


@register_model
def rith_d1_tiny_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=20,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_tiny_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=20,
        dropout=0,
        halt_pool=None,
    )


@register_model
def rith_d1_small_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_small_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_base_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=12,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_large_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1024,
        heads=16,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=24,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch4_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch4_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch8_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch8_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch8_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=8,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch16_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch16_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch32_32(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=32,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch32_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d1_huge_patch32_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=32,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=1536,
        heads=24,
        depth=1,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=32,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d3_tiny_patch4_64(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=64,
        patch_size=4,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=3,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=4,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def rith_d3_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return RiTHalt(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=3,
        halt_threshold=0.95,
        mlp_ratio=4.0,
        max_repeats=4,
        dropout=0,
        halt_pool="cls",
    )


@register_model
def srit_d1_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        repeats=50,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt=None,
        halt_noise_scale=1,
        normalize=False,
    )


@register_model
def srit_d1_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        # halt="classify",
        halt=None,
        halt_noise_scale=0,
    )


@register_model
def srit_d3_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=3,
        repeats=4,
        mlp_ratio=4.0,
        halt_threshold=0.95,
        halt="ema",
    )


@register_model
def srit_d1_base_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt="classify",
        halt_noise_scale=0,
    )


@register_model
def srit_d3_base_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=768,
        heads=12,
        depth=3,
        repeats=4,
        mlp_ratio=4.0,
        halt_threshold=0.95,
        halt="ema",
    )


@register_model
def srit2_d1_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT2(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt="classify",
        halt_noise_scale=1,
        stochastic_depth=False,
        extra_step=True,
    )

@register_model
def srit2_d1_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT2(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt="classify",
        halt_noise_scale=1,
        stochastic_depth=False,
        extra_step=False,
    )


@register_model
def srit3_d1_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT3(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt=None,
        halt_noise_scale=0,
        stochastic_depth=False,
        extra_step=False,
    )

@register_model
def srit3_d1_small_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT3(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=384,
        heads=6,
        depth=1,
        repeats=12,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt=None,
        halt_noise_scale=0,
        stochastic_depth=False,
        extra_step=False,
    )

@register_model
def srit3_d12_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained, "Pretrained models not available for this model."
    return SimpleRiT3(
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=kwargs["num_classes"],
        dim=192,
        heads=3,
        depth=12,
        repeats=1,
        mlp_ratio=4.0,
        halt_threshold=1,
        halt=None,
        halt_noise_scale=0,
        stochastic_depth=False,
        extra_step=False,
    )