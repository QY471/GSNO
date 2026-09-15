"""Strict E3 branch ablation with the complete Gaussian branch disabled.

The module retains E3's primitive and Gaussian tensors only so the formal E3
paired initial_state can be loaded with identical keys and shapes.  Those
modules are frozen and never called in forward.  The active path is therefore:

    two-stream E3 trunk -> HR fusion -> decoder -> raw bicubic base

This control measures the net contribution of E3's complete
primitive-embedding plus Gaussian-residual branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(E3GSFusion):
    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        for module in (
            self.primitive_input,
            self.primitive_residual,
            self.gaussian_refine,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.arch_summary = (
            "Strict E3-NoGaussian: E3 reconstruction trunk and decoder are "
            "unchanged; primitive embedding and Gaussian residual are frozen "
            "and bypassed"
        )

    def collect_gs_stats(self):
        return []

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

        # Deliberately do not compute primitive_input, primitive_residual, or
        # gaussian_refine.  This is the complete E3 Gaussian-branch ablation.
        residual = self.fc2(F.gelu(self.fc1(fused)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
