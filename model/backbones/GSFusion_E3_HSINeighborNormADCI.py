"""E3 with scale-stable neighbor-score normalization in LR-HSI ADCI.

Only the LR-HSI branch changes.  Before the nine-neighbor softmax, logits are
standardized across neighbors for every pixel and channel, then rescaled by a
learnable per-channel temperature.  This prevents the attention sharpness from
drifting merely because LR-HSI sampling density changes.  The HR-MSI ADCI and
the complete E3 Gaussian backend remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as CircularE3,
    compute_loss,
    sam_loss,
)


class NeighborNormADCI(ADCI):
    """ADCI whose neighbor logits have scale-stable per-channel dispersion."""

    def __init__(self, in_channels: int, mlp_hidden_dim: int) -> None:
        super().__init__(in_channels, mlp_hidden_dim)
        self.neighbor_log_scale = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = torch.chunk(self.qkv_conv(x), chunks=3, dim=1)
        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        q_minus_k = q.unsqueeze(2) - k_unfold
        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()
        scores = self.mlp(q_minus_k)

        centered = scores - scores.mean(dim=3, keepdim=True)
        neighbor_rms = centered.square().mean(dim=3, keepdim=True).add(1e-6).sqrt()
        channel_scale = self.neighbor_log_scale.exp().view(1, 1, 1, 1, -1)
        normalized_scores = centered / neighbor_rms * channel_scale
        attention = F.softmax(normalized_scores, dim=3)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * attention, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()
        return weighted_v + self.gate(x)


class GSFusion(CircularE3):
    """Formal E3 with NeighborNorm ADCI only in the LR-HSI stream."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        rng_state = torch.get_rng_state()
        self.adci_hsi_layers = nn.ModuleList(
            [NeighborNormADCI(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "Formal E3 with per-pixel/per-channel nine-neighbor score RMS "
            "normalization and learned channel temperature in HSI ADCI only; "
            "MSI ADCI and Gaussian backend unchanged"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "NeighborNormADCI", "compute_loss", "sam_loss"]
