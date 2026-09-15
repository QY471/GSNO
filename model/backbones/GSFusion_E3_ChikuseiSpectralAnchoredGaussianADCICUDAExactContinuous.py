"""Chikusei E3 with an HR spectral anchor before spatial ADCI encoding."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    GSFusion as ContinuousEllipticalGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ContinuousEllipticalGSFusion):
    """Inject bicubic HSI into the HR stream before its ADCI stack."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 128,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            max_axis_ratio=max_axis_ratio,
            **kwargs,
        )
        self.shallow_encoder2 = nn.Conv2d(num_msi + num_bands, dim, 1)
        self.arch_summary = (
            "Chikusei spectral-anchored E3; LR-HSI low-resolution ADCI stream; "
            "bicubic-HSI plus HR-MSI early-fusion HR ADCI stream; single "
            "fixed-center density-normalized constrained-elliptical Gaussian; "
            "ADCI Exact CUDA/Triton; continuous output coordinates"
        )
        self.reset_custom_init()

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
        hsi_reference = F.interpolate(
            lr_hsi, size=reference_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(torch.cat((hr_msi, hsi_reference), dim=1))
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_h = F.interpolate(f_hsi, size=reference_size, mode="bicubic", align_corners=False)
        joint = torch.cat((f_h, f_msi), dim=1)
        f_fused_reference = self.conv0(joint)
        embedding_base = self.primitive_input(joint)
        embedding = embedding_base + self.primitive_residual(embedding_base)
        gaussian_delta, gaussian_aux = self.gaussian_refine(
            embedding, reference_size, query_size, return_aux=True
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
            "HSI_reference": hsi_reference,
            "F_H": f_h,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": embedding_base,
            "E_g": embedding,
            "gaussian_delta": gaussian_delta,
            **gaussian_aux,
        }


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
