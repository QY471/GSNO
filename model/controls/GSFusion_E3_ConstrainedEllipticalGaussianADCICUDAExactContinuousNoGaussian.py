"""Exact-architecture control that bypasses the formal Gaussian residual."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    GSFusion as FormalGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(FormalGSFusion):
    """Keep all formal state keys but bypass and freeze the Gaussian branch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module in (
            self.primitive_input,
            self.primitive_residual,
            self.gaussian_refine,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.arch_summary = (
            "Formal DIM80 continuous ADCI Exact control: the complete primitive "
            "and constrained-elliptical Gaussian residual branch is frozen and "
            "bypassed; active trunk, ADCI blocks, fusion, decoder, and base are unchanged"
        )

    def collect_gs_stats(self):
        return []

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(int(value) for value in hr_msi.shape[-2:])
        query_size = reference_size if output_size is None else tuple(map(int, output_size))
        if min(query_size) <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")

        base = F.interpolate(lr_hsi, size=query_size, mode="bicubic", align_corners=False)
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        f_h = F.interpolate(
            f_hsi, size=reference_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat((f_h, f_msi), dim=1)
        f_fused_reference = self.conv0(joint)
        f_fused = (
            f_fused_reference
            if query_size == reference_size
            else F.interpolate(
                f_fused_reference,
                size=query_size,
                mode="bicubic",
                align_corners=False,
            )
        )
        residual = self.fc2(F.gelu(self.fc1(f_fused)))
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_h,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "gaussian_branch_bypassed": True,
        }


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
