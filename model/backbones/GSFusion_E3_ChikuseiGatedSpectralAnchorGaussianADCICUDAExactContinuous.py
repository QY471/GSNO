"""Chikusei E3 with a gated HR spectral anchor and optional Gaussian RMS cap."""

from __future__ import annotations

import math
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
    """Preserve the MSI stream and inject projected HSI through a small gate."""

    def __init__(
        self,
        dim: int = 96,
        num_bands: int = 128,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        spectral_anchor_initial_gate: float = 0.1,
        gaussian_rms_cap: float = 0.0,
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
        if not 0.0 < spectral_anchor_initial_gate < 1.0:
            raise ValueError("spectral_anchor_initial_gate must be in (0, 1)")
        if gaussian_rms_cap < 0.0:
            raise ValueError("gaussian_rms_cap must be non-negative")
        self.spectral_anchor_encoder = nn.Conv2d(num_bands, dim, 1)
        initial_logit = math.log(
            spectral_anchor_initial_gate / (1.0 - spectral_anchor_initial_gate)
        )
        self.spectral_anchor_logit = nn.Parameter(
            torch.full((1, dim, 1, 1), initial_logit)
        )
        self.gaussian_rms_cap = float(gaussian_rms_cap)
        self.arch_summary = (
            "Chikusei gated spectral-anchor E3; independent LR-HSI and HR-MSI "
            "ADCI Exact streams; projected bicubic HSI enters the HR stream through "
            "a per-channel sigmoid gate initialized at 0.1; optional samplewise "
            "Gaussian-to-fused RMS cap; continuous constrained-elliptical renderer"
        )
        self.reset_custom_init()

    def _bound_gaussian_delta(
        self, gaussian_delta: torch.Tensor, fused: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gaussian_rms_cap <= 0.0:
            scale = gaussian_delta.new_ones(
                gaussian_delta.shape[0], 1, 1, 1
            )
            return gaussian_delta, scale
        reduce_dims = (1, 2, 3)
        delta_rms = (
            gaussian_delta.square().mean(reduce_dims, keepdim=True) + 1e-12
        ).sqrt()
        fused_rms = (fused.square().mean(reduce_dims, keepdim=True) + 1e-12).sqrt()
        scale = (self.gaussian_rms_cap * fused_rms / delta_rms).clamp_max(1.0)
        return gaussian_delta * scale, scale

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
        f_msi_base = self.shallow_encoder2(hr_msi)
        spectral_anchor = self.spectral_anchor_encoder(hsi_reference)
        spectral_gate = torch.sigmoid(self.spectral_anchor_logit)
        f_msi = f_msi_base + spectral_gate * spectral_anchor
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
        gaussian_delta, gaussian_scale = self._bound_gaussian_delta(
            gaussian_delta, f_fused
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "HSI_reference": hsi_reference,
            "spectral_anchor": spectral_anchor,
            "spectral_gate": spectral_gate,
            "F_H": f_h,
            "F_M_base": f_msi_base,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": embedding_base,
            "E_g": embedding,
            "gaussian_delta": gaussian_delta,
            "gaussian_rms_scale": gaussian_scale,
            **gaussian_aux,
        }


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
