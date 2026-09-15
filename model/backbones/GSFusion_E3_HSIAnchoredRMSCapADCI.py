"""E3 with training-anchored RMS caps in the LR-HSI ADCI stream.

The original ADCI scores are preserved unless a pixel/channel neighbor-score
RMS exceeds a fixed multiple of that channel's running 4x training RMS.  The
running statistic is learned only from training batches and reused unchanged
at evaluation, so high-scale inputs cannot redefine their own normalization.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as CircularE3,
    compute_loss,
    sam_loss,
)


class TrainingAnchoredRMSCapADCI(ADCI):
    """ADCI that only attenuates scores outside the 4x training RMS range."""

    def __init__(
        self,
        in_channels: int,
        mlp_hidden_dim: int,
        cap_multiplier: float = 2.0,
        momentum: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(in_channels, mlp_hidden_dim)
        if cap_multiplier <= 0:
            raise ValueError("cap_multiplier must be positive")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")
        self.cap_multiplier = float(cap_multiplier)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_score_rms", torch.ones(in_channels))
        self.register_buffer(
            "num_batches_tracked", torch.zeros((), dtype=torch.long)
        )
        self.last_stats: Dict[str, float] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query, key, value = torch.chunk(self.qkv_conv(x), chunks=3, dim=1)
        key_neighbors = F.unfold(key, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        value_neighbors = F.unfold(value, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        query_key_difference = (
            query.unsqueeze(2) - key_neighbors
        ).permute(0, 3, 4, 2, 1).contiguous()
        raw_scores = self.mlp(query_key_difference)
        centered_scores = raw_scores - raw_scores.mean(dim=3, keepdim=True)
        local_score_rms = centered_scores.square().mean(dim=3).add(self.eps).sqrt()

        if self.training:
            with torch.no_grad():
                batch_score_rms = centered_scores.detach().square().mean(
                    dim=(0, 1, 2, 3)
                ).add(self.eps).sqrt()
                ema_score_rms = self.running_score_rms.lerp(
                    batch_score_rms, self.momentum
                )
                updated_score_rms = torch.where(
                    self.num_batches_tracked == 0,
                    batch_score_rms,
                    ema_score_rms,
                )
                self.running_score_rms.copy_(updated_score_rms)
                self.num_batches_tracked.add_(1)
            reference_rms = updated_score_rms
        else:
            reference_rms = self.running_score_rms.clamp_min(self.eps)

        cap = reference_rms.view(1, 1, 1, channels) * self.cap_multiplier
        attenuation = (cap / local_score_rms.clamp_min(self.eps)).clamp(max=1.0)
        effective_scores = centered_scores * attenuation.unsqueeze(3)
        attention = F.softmax(effective_scores, dim=3)

        neighbors = value_neighbors.permute(0, 3, 4, 2, 1).contiguous()
        weighted_value = torch.sum(neighbors * attention, dim=3)
        weighted_value = weighted_value.permute(0, 3, 1, 2).contiguous()
        output = weighted_value + self.gate(x)

        if not self.training:
            with torch.no_grad():
                self.last_stats = {
                    "score_rms_mean": float(local_score_rms.mean()),
                    "running_score_rms_mean": float(self.running_score_rms.mean()),
                    "running_score_rms_min": float(self.running_score_rms.min()),
                    "running_score_rms_max": float(self.running_score_rms.max()),
                    "cap_multiplier": self.cap_multiplier,
                    "clipped_fraction": float((attenuation < 1.0).float().mean()),
                    "attenuation_mean": float(attenuation.mean()),
                    "attenuation_min": float(attenuation.min()),
                }
        return output


class GSFusion(CircularE3):
    """Formal E3 with anchored RMS caps only in the LR-HSI ADCI layers."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        adci_rms_cap_multiplier: float = 2.0,
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
            [
                TrainingAnchoredRMSCapADCI(
                    dim,
                    dim,
                    cap_multiplier=adci_rms_cap_multiplier,
                )
                for _ in range(adci_layers)
            ]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "Formal E3 with per-channel HSI ADCI score caps anchored to running "
            "4x training RMS; below-cap scores, MSI ADCI, and Gaussian backend "
            "remain unchanged"
        )
        self.reset_custom_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = super().collect_gs_stats()
        for layer_index, layer in enumerate(self.adci_hsi_layers, start=1):
            if layer.last_stats is not None:
                stats.append(
                    {
                        "layer": f"hsi_rmscap_{layer_index}",
                        **layer.last_stats,
                    }
                )
        return stats


__all__ = [
    "GSFusion",
    "TrainingAnchoredRMSCapADCI",
    "compute_loss",
    "sam_loss",
]
