"""DIM-configurable E3 with parallel constrained elliptical Gaussian experts."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_E3_ThreeConstrainedEllipticalGaussian import (
    GSFusion as SequentialThreeGaussianGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(SequentialThreeGaussianGSFusion):
    """Mix three Gaussian residuals that share the same primitive input."""

    variant_name = "three_parallel_constrained_elliptical_gaussian_experts"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussian_mixture_logits = nn.Parameter(torch.zeros(3))
        self.arch_summary = (
            "E3 DIM-configurable three-expert parallel constrained elliptical "
            "Gaussian mixture: every expert reads the same primitive field; "
            "softmax-normalized global mixing; no sequential state cascade"
        )

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        weights = torch.softmax(self.gaussian_mixture_logits.detach(), dim=0)
        layers = [self.gaussian_refine, *self.gaussian_refine_extra]
        result = []
        for index, (layer, weight) in enumerate(zip(layers, weights), start=1):
            if layer.last_stats is not None:
                result.append(
                    {
                        "layer": f"hr_gaussian_parallel_{index}",
                        "parallel_mixture_weight": float(weight),
                        **layer.last_stats,
                    }
                )
        return result

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)

        gaussian_layers = [self.gaussian_refine, *self.gaussian_refine_extra]
        deltas = [layer(primitive) - primitive for layer in gaussian_layers]
        weights = torch.softmax(self.gaussian_mixture_logits, dim=0)
        gaussian_delta = torch.stack(deltas, dim=0).mul(
            weights.view(-1, 1, 1, 1, 1)
        ).sum(dim=0)

        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
