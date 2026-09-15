"""GSFusion Baseline EDSR backbone with the current normalized 3-sigma scatter.

This controlled experiment preserves the input/feature extractor of
``GSFusion_Baseline.py``:

    concat(HR-MSI, bicubic LR-HSI) -> EDSR(6 residual blocks)

It replaces the legacy three-layer raw ScatterGS stack with exactly one
pixel-centered circular Gaussian residual using density normalization and the
adaptive 3-sigma CUDA rasterizer.  The decoder and zero-residual
initialization follow the current S0 protocol.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.support.EDSR import make_edsr_baseline
from model.geometry.GSFusion_HRFused_AdaptiveGaussianResidual_Isotropic import (
    HRAdaptiveGaussianResidual,
    compute_loss,
    sam_loss,
)


class GSFusion(nn.Module):
    """Baseline convolutional encoder plus current circular Gaussian residual."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        n_resblocks: int = 6,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)
        self.n_resblocks = int(n_resblocks)

        self.edsr_encoder = make_edsr_baseline(
            n_resblocks=self.n_resblocks,
            n_feats=self.dim,
            n_colors=self.num_bands + self.num_msi,
            no_upsampling=True,
        )
        self.gaussian_refine = HRAdaptiveGaussianResidual(self.dim)
        self.fc1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.fc2 = nn.Conv2d(self.dim, self.num_bands, kernel_size=1)

        self.arch_summary = (
            "raw bicubic HSI + HR-MSI concat -> EDSR 3x3 encoder "
            f"({self.n_resblocks} residual blocks) -> one density-normalized "
            "adaptive-3sigma circular CUDA Gaussian residual -> "
            "1x1/GELU/1x1 decoder -> raw bicubic residual base; "
            "fixed offset=0/rho=0, learned scalar std/opacity/value"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        """Start from the raw bicubic base, matching the current S0 protocol."""
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.gaussian_refine.last_stats
        return [] if stats is None else [{"layer": "hr_gaussian", **stats}]

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
    ) -> torch.Tensor:
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        # Keep the exact raw-input order used by GSFusion_Baseline.py.
        joint = torch.cat([hr_msi, base], dim=1)
        fused = self.edsr_encoder(joint)
        refined = self.gaussian_refine(fused)
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "HRAdaptiveGaussianResidual",
    "compute_loss",
    "sam_loss",
]
