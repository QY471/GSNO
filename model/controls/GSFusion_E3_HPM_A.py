"""E3-HPM-A branch-localization diagnostic.

Only the HSI encoder is replaced by channel-only pointwise gated residual
blocks.  The MSI branch keeps the original three ADCI blocks.  Fusion,
primitive embedding, circular density-normalized adaptive-3-sigma Gaussian
rendering, decoder, and the raw bicubic HSI residual base are unchanged from
the formal E3 model.

This is a diagnostic model, not a claimed final architecture.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    HRAdaptiveGaussianResidual,
)


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class PointwiseGatedBlock(nn.Module):
    """Channel-only residual block; it never reads spatial neighbours."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = int(dim) * int(expansion)
        self.norm = ChannelLayerNorm(dim)
        self.in_proj = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        update = self.out_proj(F.gelu(value) * torch.sigmoid(gate))
        return x + update


class GSFusion(nn.Module):
    """Pointwise HSI encoder + original ADCI MSI encoder + unchanged E3."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        pointwise_expansion: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.encoder_block_count = int(adci_layers)

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, kernel_size=1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, kernel_size=1)
        self.hsi_pointwise_layers = nn.ModuleList(
            [
                PointwiseGatedBlock(dim, expansion=pointwise_expansion)
                for _ in range(adci_layers)
            ]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )

        # Everything below is the formal E3 backend.
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.gaussian_refine = HRAdaptiveGaussianResidual(dim)
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc2 = nn.Conv2d(dim, num_bands, kernel_size=1)
        self.primitive_input = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.arch_summary = (
            "E3-HPM-A branch-localization diagnostic: HSI=Conv1x1 plus "
            "three channel-only PointwiseGatedBlocks; MSI=original ADCI x3; "
            "formal E3 fusion, primitive embedding, circular normalized "
            "adaptive-3sigma Gaussian residual, decoder, and bicubic base"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats: Optional[Dict[str, float]] = self.gaussian_refine.last_stats
        return [] if stats is None else [{"layer": "hr_gaussian", **stats}]

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        for block in self.hsi_pointwise_layers:
            f_hsi = block(f_hsi)

        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_h = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_m = f_msi
        joint = torch.cat([f_h, f_m], dim=1)

        f_fused = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        primitive_with_delta = self.gaussian_refine(e_g)
        gaussian_delta = primitive_with_delta - e_g
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
