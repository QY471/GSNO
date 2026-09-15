"""Scale-stable E3 primitive content with bounded cross-modal factorization.

The proven E3 HR primitive embedding is retained as a stable base.  A new
scale-stable residual uses HSI features for spectral direction and fixed-HR
MSI local contrast for spatial modulation.  Unlike the old 52.79-dB LR-cell
transport, it never flattens a ratio-dependent HR region into a fixed 4x4
footprint.  The residual is zero initialized and RMS-capped so it cannot
overwhelm the E3 base at unseen input ratios.  Gaussian rendering is performed
in native HR-MSI reference coordinates and supports arbitrary output_size.

The model is kept in its own file so the formal comparison changes only this
primitive-content branch and the continuous reference-coordinate renderer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_E3_HRPrimitiveReferenceContinuous import (
    GSFusion as E3ReferenceContinuousControl,
)
from model.GSFusion_GSNO import compute_loss, sam_loss


class GSFusion(E3ReferenceContinuousControl):
    """E3 plus a bounded HR spectral-spatial primitive residual."""

    def __init__(
        self,
        dim: int = 64,
        factorized_cap_ratio: float = 0.25,
        **kwargs: object,
    ) -> None:
        if not 0.0 < factorized_cap_ratio <= 1.0:
            raise ValueError("factorized_cap_ratio must lie in (0, 1]")
        self.factorized_cap_ratio = float(factorized_cap_ratio)
        super().__init__(dim=dim, **kwargs)

        # All learned transforms are pointwise. Spatial scale comes only from
        # the fixed 5x5 HR-pixel reference, so its meaning does not change when
        # the LR-HSI/HR-MSI input ratio changes.
        self.hsi_spectral_direction = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.msi_local_gate = nn.Conv2d(dim, dim, 1, bias=False)
        self.factorized_out = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.factorized_out.weight)
        nn.init.zeros_(self.factorized_out.bias)

        self.arch_summary = (
            "E3 HR Factorized Reference Continuous: stable E3 full-HR "
            "primitive embedding plus zero-initialized, per-pixel RMS-capped "
            "HSI spectral direction times fixed-HR MSI local-contrast gate; "
            "no LR footprint and no E6 routing; stride-1 circular normalized "
            "adaptive-3sigma rendering in native HR-MSI coordinates; optional "
            "continuous output_size"
        )

    def reset_custom_init(self) -> None:
        # super().__init__ calls this method before the new layers exist.
        super().reset_custom_init()
        if hasattr(self, "factorized_out"):
            nn.init.zeros_(self.factorized_out.weight)
            nn.init.zeros_(self.factorized_out.bias)

    @staticmethod
    def fixed_hr_local_reference(feature: torch.Tensor) -> torch.Tensor:
        padded = F.pad(feature, (2, 2, 2, 2), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=5, stride=1)

    def _bounded_factorized_residual(
        self,
        e_base: torch.Tensor,
        f_h: torch.Tensor,
        f_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        spectral_direction = self.hsi_spectral_direction(f_h)
        msi_local_detail = f_m - self.fixed_hr_local_reference(f_m)
        spatial_gate = torch.tanh(self.msi_local_gate(msi_local_detail))
        raw_residual = self.factorized_out(spectral_direction * spatial_gate)

        raw_rms = raw_residual.square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
        # Detaching the reference prevents the new branch from increasing its
        # allowance by inflating the stable E3 embedding itself.
        base_rms = (
            e_base.detach().square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
        )
        allowed_rms = self.factorized_cap_ratio * base_rms
        cap_scale = torch.clamp(allowed_rms / raw_rms, max=1.0)
        bounded_residual = raw_residual * cap_scale
        return bounded_residual, {
            "spectral_direction": spectral_direction,
            "msi_local_detail": msi_local_detail,
            "spatial_gate": spatial_gate,
            "factorized_raw": raw_residual,
            "factorized_cap_scale": cap_scale,
        }

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
        query_size = (
            reference_size
            if output_size is None
            else (int(output_size[0]), int(output_size[1]))
        )
        if query_size[0] <= 0 or query_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")

        base = F.interpolate(
            lr_hsi, size=query_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_h = F.interpolate(
            f_hsi, size=reference_size, mode="bicubic", align_corners=False
        )
        f_m = f_msi
        joint = torch.cat((f_h, f_m), dim=1)
        f_fused_reference = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_base = e0 + self.primitive_residual(e0)
        factorized_delta, factorized_aux = self._bounded_factorized_residual(
            e_base, f_h, f_m
        )
        e_g = e_base + factorized_delta

        gaussian_delta, gaussian_aux = self.gaussian_transport(
            e_g,
            reference_size=reference_size,
            out_size=query_size,
            return_aux=True,
        )
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
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            raw_rms = factorized_aux["factorized_raw"].square().mean().sqrt()
            bounded_rms = factorized_delta.square().mean().sqrt()
            e_base_rms = e_base.square().mean().sqrt()
            cap_scale = factorized_aux["factorized_cap_scale"]
            self._last_stats = {
                "e_base_abs_mean": float(e_base.detach().abs().mean()),
                "factorized_raw_rms": float(raw_rms.detach()),
                "factorized_bounded_rms": float(bounded_rms.detach()),
                "factorized_to_base_rms_ratio": float(
                    bounded_rms.detach() / (e_base_rms.detach() + 1e-8)
                ),
                "factorized_cap_active_ratio": float(
                    (cap_scale.detach() < 0.999999).float().mean()
                ),
                "spatial_gate_abs_mean": float(
                    factorized_aux["spatial_gate"].detach().abs().mean()
                ),
                "gaussian_delta_abs_mean": float(
                    gaussian_delta.detach().abs().mean()
                ),
                "gaussian_to_fused_ratio": float(
                    gaussian_delta.detach().abs().mean()
                    / (f_fused.detach().abs().mean() + 1e-8)
                ),
                "reference_height": float(reference_size[0]),
                "reference_width": float(reference_size[1]),
                "query_height": float(query_size[0]),
                "query_width": float(query_size[1]),
            }

        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_h,
            "F_M": f_m,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": e0,
            "E_base": e_base,
            "factorized_delta": factorized_delta,
            "E_g": e_g,
            "gaussian_delta": gaussian_delta,
            **factorized_aux,
            **gaussian_aux,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        return [] if not stats else [{"layer": "e3_hr_factorized_field", **stats}]


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
