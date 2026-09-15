"""E3 with physical-span correction in the LR-HSI ADCI branch.

The correction is parameter-free and exactly inactive at the 4x training
scale.  At larger scale factors one LR-HSI neighbor represents a proportionally
larger HR distance, so off-center attention mass is contracted by 4/sf and the
removed mass is reassigned to the center.  Constant-field preservation and a
unit attention sum are retained.  The HR-MSI branch and Gaussian backend are
unchanged.
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


class PhysicalSpanADCI(ADCI):
    """Legacy ADCI with parameter-free physical-span neighbor contraction."""

    def forward(
        self,
        x: torch.Tensor,
        physical_scale: float = 4.0,
        reference_scale: float = 4.0,
    ) -> torch.Tensor:
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
        weights = F.softmax(scores, dim=3)

        contraction = min(1.0, float(reference_scale) / float(physical_scale))
        if contraction < 1.0:
            center = weights[:, :, :, 4, :]
            off_mass = weights.sum(dim=3) - center
            adjusted_center = center + (1.0 - contraction) * off_mass
            contracted = weights * contraction
            weights = torch.cat(
                [
                    contracted[:, :, :, :4, :],
                    adjusted_center.unsqueeze(3),
                    contracted[:, :, :, 5:, :],
                ],
                dim=3,
            )

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * weights, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()
        return weighted_v + self.gate(x)


class GSFusion(CircularE3):
    """Formal E3 with physical-span correction only in LR-HSI ADCI."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        reference_scale: float = 4.0,
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
            [PhysicalSpanADCI(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)
        self.reference_scale = float(reference_scale)
        self.arch_summary = (
            "Formal E3 with parameter-free LR-HSI ADCI physical-span "
            "correction: off-center mass scales by reference_sf/current_sf "
            "and removed mass returns to center; MSI ADCI and Gaussian path "
            "unchanged"
        )
        self.reset_custom_init()

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        target_size = hr_msi.shape[-2:]
        inferred_scale = float(target_size[-1]) / float(lr_hsi.shape[-1])
        physical_scale = inferred_scale if sf is None else float(sf)
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(
                f_hsi,
                physical_scale=physical_scale,
                reference_scale=self.reference_scale,
            )
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)
        gaussian_delta = self.gaussian_refine(primitive) - primitive
        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "PhysicalSpanADCI", "compute_loss", "sam_loss"]
