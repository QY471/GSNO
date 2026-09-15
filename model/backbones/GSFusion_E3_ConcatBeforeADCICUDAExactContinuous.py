"""E3 ablation that fuses the two HR latents before shared ADCI processing."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.GSFusion_GSNO import compute_loss, sam_loss
from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    GSFusion as ContinuousBase,
)


class GSFusion(ContinuousBase):
    """Concat HR latents first, then apply the same six ADCI blocks serially."""

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(int(v) for v in hr_msi.shape[-2:])
        query_size = reference_size if output_size is None else tuple(map(int, output_size))
        if min(query_size) <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")

        base = F.interpolate(lr_hsi, size=query_size, mode="bicubic", align_corners=False)
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_hsi = F.interpolate(f_hsi, size=reference_size, mode="bicubic", align_corners=False)
        f_msi = self.shallow_encoder2(hr_msi)
        joint = torch.cat((f_hsi, f_msi), dim=1)
        f_fused_reference = self.conv0(joint)
        for layer in self.adci_hsi_layers:
            f_fused_reference = layer(f_fused_reference)
        for layer in self.adci_msi_layers:
            f_fused_reference = layer(f_fused_reference)

        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        gaussian_delta, gaussian_aux = self.gaussian_refine(
            e_g, reference_size, query_size, return_aux=True
        )
        f_fused = f_fused_reference if query_size == reference_size else F.interpolate(
            f_fused_reference, size=query_size, mode="bicubic", align_corners=False
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_hsi,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": e0,
            "E_g": e_g,
            "gaussian_delta": gaussian_delta,
            **gaussian_aux,
        }


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
