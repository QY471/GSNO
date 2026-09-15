"""Checkpoint-compatible exact optimization of the adopted ADCI block."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import LayerNorm


def _resolve_exact_aggregate():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_root = os.path.join(repo_root, "extensions", "adci_exact_triton")
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from adci_exact_triton import adci_exact_aggregate

    return adci_exact_aggregate


class ADCIExact(nn.Module):
    """ADCI with the original score path and fused CUDA value aggregation.

    Parameter names and tensor shapes intentionally match ``GSFusion_GSNO.ADCI``
    so existing state dictionaries load with ``strict=True``.
    """

    def __init__(self, in_channels: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        if int(in_channels) != int(mlp_hidden_dim):
            raise ValueError(
                "ADCIExact currently requires mlp_hidden_dim == in_channels"
            )
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
        self._aggregate = _resolve_exact_aggregate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = torch.chunk(self.qkv_conv(x), chunks=3, dim=1)

        # Keep the legacy score path unchanged.  Although W(q-k)=Wq-Wk in
        # exact arithmetic, factorizing it changes float32 accumulation order
        # enough to measurably perturb gradients.  The CUDA optimization below
        # therefore targets only the expensive value unfold/reduction.
        k_neighbors = F.unfold(k, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        hidden = q.unsqueeze(2) - k_neighbors
        hidden = hidden.permute(0, 3, 4, 2, 1).contiguous()
        scores = self.mlp(hidden).contiguous()

        weighted_v = self._aggregate(scores, v.contiguous(), use_triton=True)
        return weighted_v + self.gate(x)


__all__ = ["ADCIExact"]
