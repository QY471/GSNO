"""Pure PyTorch implementation of the checkpoint-compatible ADCI block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import LayerNorm


class ADCIPyTorch(nn.Module):
    """ADCI with native unfold, softmax, and weighted value aggregation."""

    def __init__(self, in_channels: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.qkv_conv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = torch.chunk(self.qkv_conv(x), chunks=3, dim=1)
        k_neighbors = F.unfold(k, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        v_neighbors = F.unfold(v, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        hidden = q.unsqueeze(2) - k_neighbors
        hidden = hidden.permute(0, 3, 4, 2, 1).contiguous()
        scores = self.mlp(hidden)
        weights = F.softmax(scores, dim=3)
        neighbors = v_neighbors.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors * weights, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()
        return weighted_v + self.gate(x)


__all__ = ["ADCIPyTorch"]
