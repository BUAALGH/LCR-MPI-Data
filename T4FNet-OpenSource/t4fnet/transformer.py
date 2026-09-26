from dataclasses import dataclass

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


@dataclass
class TransformerConfig:
    hidden_size: int = 256
    num_heads: int = 4
    mlp_dim: int = 512
    blocks_per_stage: tuple = (2, 2, 2)
    decoder_dim: int = 64
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    drop_path_rate: float = 0.1
    qmod_scale_init: float = 0.8


class SoftMaskGlobalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_drop=0.0, proj_drop=0.0,
                 vis=False, qmod_scale_init=0.8):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.bias_scale = nn.Parameter(torch.ones(1))
        self.qmod_scale = nn.Parameter(torch.tensor(float(qmod_scale_init)))

    def _split_heads(self, x):
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x, soft_mask=None):
        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))
        if soft_mask is not None:
            weight = soft_mask.unsqueeze(1).unsqueeze(-1)
            q = q * (1.0 + weight * self.qmod_scale.clamp(0.0, 2.0))
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = self.softmax(attention)
        weights = attention if self.vis else None
        attention = self.attn_drop(attention)
        output = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous()
        batch, tokens, _, _ = output.shape
        output = output.view(batch, tokens, self.num_heads * self.head_dim)
        return self.proj_drop(self.out(output)), weights


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SATransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, attn_drop=0.0,
                 drop=0.0, drop_path=0.0, vis=False, qmod_scale_init=0.8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = SoftMaskGlobalAttention(
            hidden_size, num_heads, attn_drop, drop, vis, qmod_scale_init
        )
        self.ffn = Mlp(hidden_size, mlp_dim, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, soft_mask=None):
        attention, weights = self.attn(self.norm1(x), soft_mask)
        x = x + self.drop_path(attention)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x, weights


class SATransLevel(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, num_blocks,
                 attn_drop=0.0, drop=0.0, drop_paths=None, vis=False,
                 qmod_scale_init=0.8):
        super().__init__()
        drop_paths = drop_paths or [0.0] * num_blocks
        self.blocks = nn.ModuleList([
            SATransBlock(
                hidden_size, num_heads, mlp_dim, attn_drop, drop,
                drop_paths[index], vis, qmod_scale_init
            )
            for index in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.skip_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, soft_mask=None, prev_skip=None, return_attn=False):
        if prev_skip is not None:
            x = x + self.skip_proj(prev_skip)
        weights = [] if return_attn else None
        for block in self.blocks:
            x, block_weights = block(x, soft_mask)
            if return_attn:
                weights.append(block_weights)
        output = self.norm(x)
        return (output, weights) if return_attn else output


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size=4, img_size=64, drop=0.0):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.img_size = img_size
        patch_count = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"Expected [B, C, {self.img_size}, {self.img_size}]")
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.drop(x + self.pos_embed), height, width


class SATransEncoderV4(nn.Module):
    def __init__(self, config, vis=True):
        super().__init__()
        blocks = config.blocks_per_stage
        total = sum(blocks[:2])
        drop_paths = [value.item() for value in torch.linspace(0, config.drop_path_rate, total)]
        common = (
            config.hidden_size,
            config.num_heads,
            config.mlp_dim,
        )
        self.level1 = SATransLevel(
            *common, blocks[0], config.attention_dropout_rate,
            config.dropout_rate, drop_paths[:blocks[0]], vis,
            config.qmod_scale_init,
        )
        self.level2 = SATransLevel(
            *common, blocks[1], config.attention_dropout_rate,
            config.dropout_rate, drop_paths[blocks[0]:], vis,
            config.qmod_scale_init,
        )

    def forward(self, x, soft_mask=None, return_attn=False):
        if return_attn:
            skip1, weights1 = self.level1(
                x, soft_mask, prev_skip=None, return_attn=True
            )
            skip2, weights2 = self.level2(
                skip1, soft_mask, prev_skip=skip1, return_attn=True
            )
            return [skip1, skip2], [weights1, weights2]
        skip1 = self.level1(x, soft_mask, prev_skip=None)
        skip2 = self.level2(skip1, soft_mask, prev_skip=skip1)
        return [skip1, skip2]

