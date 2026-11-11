"""PatchSTG adaptation using explicit PyTorch attention; see third-party notices.

Reference: LMissher/PatchSTG, commit 8b298b230227a304a082e05241c69dca3740a73d.
This is a code-aligned adaptation for the shared data protocol, not an official
reproduction. No timm dependency, pretrained weights, or source training runner.
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from traffic_benchmark.data.spatial import SpatialMaps


@dataclass(frozen=True)
class PatchSTGConfig:
    nodes: int = 716
    history: int = 12
    signal_dim: int = 64
    node_dim: int = 64
    slot_dim: int = 32
    weekday_dim: int = 32
    slots_per_day: int = 96
    layers: int = 5
    dropout: float = 0.1

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if name != "dropout" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")

    @property
    def width(self):
        return self.signal_dim + self.node_dim + self.slot_dim + self.weekday_dim


class SpatialAttention(nn.Module):
    """One-head scaled dot-product attention with biased QKV and output layers."""

    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.proj = nn.Linear(width, width, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = width ** -0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        weights = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj_drop(self.proj(self.attn_drop(weights) @ v))


class FeedForward(nn.Module):
    """Reference MLP ratio one: linear, GELU, dropout, linear, dropout."""

    def __init__(self, width, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.act, self.drop1, self.drop2 = nn.GELU(), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class DualAttentionBlock(nn.Module):
    """Depth: positions within each patch; breadth: same position across patches."""

    def __init__(self, width, patch_count, patch_size, dropout=0.1):
        super().__init__()
        self.patch_count, self.patch_size = patch_count, patch_size
        self.depth_norm1, self.depth_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.breadth_norm1, self.breadth_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.depth_attention, self.breadth_attention = SpatialAttention(width, dropout), SpatialAttention(width, dropout)
        self.depth_mlp, self.breadth_mlp = FeedForward(width, dropout), FeedForward(width, dropout)

    def forward(self, x):
        batch, time, nodes, width = x.shape
        if nodes != self.patch_count * self.patch_size:
            raise ValueError("Padded sensor count differs from block patch layout.")
        grid = x.reshape(batch, time, self.patch_count, self.patch_size, width)
        depth = self.depth_norm1(grid).reshape(-1, self.patch_size, width)
        grid = grid + self.depth_attention(depth).reshape_as(grid)
        grid = grid + self.depth_mlp(self.depth_norm2(grid))
        breadth = self.breadth_norm1(grid).transpose(2, 3).reshape(-1, self.patch_count, width)
        update = self.breadth_attention(breadth).reshape(batch, time, self.patch_size, self.patch_count, width).transpose(2, 3)
        grid = grid + update
        grid = grid + self.breadth_mlp(self.breadth_norm2(grid))
        return grid.reshape(batch, time, nodes, width)


class PatchSTG(nn.Module):
    """One temporal patch (the reference SD path), H-step normalized prediction.

    Input x: (B,H,N,1); calendar: integer (B,H,2) or (B,H,N,2).
    Output: (B,H,N,1). Caller inverse-transforms before original-unit evaluation.
    """

    def __init__(self, config: PatchSTGConfig, maps: SpatialMaps):
        super().__init__()
        maps.validate(config.nodes)
        self.config = config
        self.register_buffer("gather_indices", torch.tensor(np.array(maps.gather), dtype=torch.long))
        self.register_buffer("restore_indices", torch.tensor(np.array(maps.restore), dtype=torch.long))
        self.signal_projection = nn.Conv2d(3, config.signal_dim, kernel_size=(1, config.history), stride=(1, config.history))
        self.node_embedding = nn.Parameter(torch.empty(config.nodes, config.node_dim))
        self.slot_embedding = nn.Parameter(torch.empty(config.slots_per_day, config.slot_dim))
        self.weekday_embedding = nn.Parameter(torch.empty(7, config.weekday_dim))
        for embedding in (self.node_embedding, self.slot_embedding, self.weekday_embedding):
            nn.init.xavier_uniform_(embedding)
        self.blocks = nn.ModuleList([DualAttentionBlock(config.width, maps.patch_count, maps.patch_size, config.dropout)
                                     for _ in range(config.layers)])
        self.head = nn.Conv2d(config.width, config.history, kernel_size=1)

    def forward(self, x, calendar):
        cfg = self.config
        if x.ndim != 4 or x.shape[1:] != (cfg.history, cfg.nodes, 1) or x.shape[0] < 1:
            raise ValueError("Traffic shape must be (batch,history,nodes,1).")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("Traffic must be finite float32 normalized values.")
        if calendar.ndim == 3 and calendar.shape == (x.shape[0], cfg.history, 2):
            calendar = calendar.unsqueeze(2).expand(-1, -1, cfg.nodes, -1)
        if calendar.shape != (x.shape[0], cfg.history, cfg.nodes, 2) or calendar.dtype != torch.long:
            raise ValueError("Calendar must be int64 (B,H,2) or (B,H,N,2).")
        if calendar.device != x.device or self.gather_indices.device != x.device:
            raise ValueError("Model, traffic and calendar must be on the same device.")
        slots, weekdays = calendar[..., 0], calendar[..., 1]
        if ((slots < 0) | (slots >= cfg.slots_per_day) | (weekdays < 0) | (weekdays >= 7)).any():
            raise ValueError("Calendar values out of range.")
        scalars = torch.stack((slots / cfg.slots_per_day, weekdays / 7), dim=-1)
        signal = torch.cat((x, scalars), dim=-1).transpose(1, 3)
        hidden = self.signal_projection(signal).transpose(1, 3)
        # Reference SD uses the last input time's embeddings, not target calendar.
        hidden = torch.cat((hidden, self.slot_embedding[slots[:, -1:]],
                            self.weekday_embedding[weekdays[:, -1:]],
                            self.node_embedding[None, None].expand(x.shape[0], 1, -1, -1)), dim=-1)
        padded = hidden.index_select(2, self.gather_indices)
        for block in self.blocks:
            padded = block(padded)
        restored = padded.index_select(2, self.restore_indices)
        return self.head(restored.transpose(2, 3).reshape(x.shape[0], cfg.width, cfg.nodes, 1))
