"""EDSR reconstruction backbone with a dedicated Gaussian primitive embedding.

The reconstruction path is identical to the paired direct-head experiment:

    J = concat(HR-MSI, bicubic LR-HSI)
    F = EDSR(J)

A separate pointwise branch maps the same pre-EDSR input J to E_g.  Gaussian
value, scalar circular std, and opacity are predicted only from E_g.  The
density-normalized adaptive-3sigma Gaussian delta is added to F before the
unchanged decoder.
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
    """Paired EDSR experiment whose Gaussian heads read a dedicated E_g."""

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
        joint_channels = self.num_bands + self.num_msi

        # Common modules are registered in exactly the same order as the
        # direct-head EDSR experiment, preserving their seeded Xavier stream.
        self.edsr_encoder = make_edsr_baseline(
            n_resblocks=self.n_resblocks,
            n_feats=self.dim,
            n_colors=joint_channels,
            no_upsampling=True,
        )
        self.gaussian_refine = HRAdaptiveGaussianResidual(self.dim)
        self.fc1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.fc2 = nn.Conv2d(self.dim, self.num_bands, kernel_size=1)

        # Experiment-only modules come after all shared modules.  Restore the
        # constructor RNG so their allocation does not alter external seeded
        # state before the training script performs Xavier initialization.
        extra_rng_state = torch.get_rng_state()
        self.primitive_input = nn.Conv2d(joint_channels, self.dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1),
        )
        torch.set_rng_state(extra_rng_state)

        self.arch_summary = (
            "J=concat(raw HR-MSI,bicubic HSI); F=EDSR(J) with 6 residual "
            "blocks; E0=Conv1x1(J); "
            "E_g=E0+Conv1x1(GELU(Conv1x1(E0))); Gaussian heads read E_g; "
            "one density-normalized adaptive-3sigma circular CUDA delta is "
            "added to F; unchanged 1x1/GELU/1x1 decoder and bicubic base"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)

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
        joint = torch.cat([hr_msi, base], dim=1)

        fused = self.edsr_encoder(joint)
        E0 = self.primitive_input(joint)
        E_g = E0 + self.primitive_residual(E0)
        primitive_with_delta = self.gaussian_refine(E_g)
        gaussian_delta = primitive_with_delta - E_g

        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "HRAdaptiveGaussianResidual",
    "compute_loss",
    "sam_loss",
]
