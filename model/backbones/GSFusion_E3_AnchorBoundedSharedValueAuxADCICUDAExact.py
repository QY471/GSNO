"""Anchor-preserving E3 with one bounded shared-value auxiliary Gaussian."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.GSFusion_E3_ParallelTwoSharedValueADCICUDAExact import (
    GSFusion as ParallelTwoSharedValueGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ParallelTwoSharedValueGSFusion):
    """Keep the first Gaussian intact and bound the second expert's correction."""

    variant_name = "anchor_bounded_shared_value_aux_gaussian_adci_cuda_exact"

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        auxiliary_beta_max: float = 0.25,
        correction_rms_cap_ratio: float = 0.25,
        auxiliary_gate_init_logit: float = -6.0,
        **kwargs: object,
    ) -> None:
        if not 0.0 < auxiliary_beta_max <= 1.0:
            raise ValueError("auxiliary_beta_max must lie in (0, 1]")
        if not 0.0 < correction_rms_cap_ratio <= 1.0:
            raise ValueError("correction_rms_cap_ratio must lie in (0, 1]")
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            max_axis_ratio=max_axis_ratio,
            **kwargs,
        )
        del self.gaussian_mixture_logits
        self.auxiliary_beta_max = float(auxiliary_beta_max)
        self.correction_rms_cap_ratio = float(correction_rms_cap_ratio)
        self.auxiliary_gate_logit = nn.Parameter(
            torch.tensor(float(auxiliary_gate_init_logit))
        )
        self._anchor_stats: Dict[str, float] = {}
        self.arch_summary = (
            "E3 DIM80 ADCI Exact with one complete constrained-elliptical "
            "Gaussian anchor and one parallel shared-value geometry expert; "
            "the learned auxiliary difference has beta <= 0.25 and its RMS "
            "is capped at 0.25 of the anchor RMS; no sf input and no cascade"
        )

    def _bound_auxiliary_correction(
        self,
        anchor_delta: torch.Tensor,
        auxiliary_delta: torch.Tensor,
    ):
        beta = self.auxiliary_beta_max * torch.sigmoid(
            self.auxiliary_gate_logit
        )
        raw_correction = beta * (auxiliary_delta - anchor_delta)
        reduce_dims = tuple(range(1, raw_correction.ndim))
        anchor_rms = anchor_delta.square().mean(
            dim=reduce_dims, keepdim=True
        ).add(1e-12).sqrt()
        correction_rms = raw_correction.square().mean(
            dim=reduce_dims, keepdim=True
        ).add(1e-12).sqrt()
        allowed_rms = self.correction_rms_cap_ratio * anchor_rms
        cap_scale = torch.clamp(allowed_rms / correction_rms, max=1.0)
        return raw_correction * cap_scale, beta, cap_scale

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        result = []
        if self.gaussian_refine.last_stats is not None:
            result.append(
                {
                    "layer": "hr_gaussian_anchor",
                    **self.gaussian_refine.last_stats,
                    **self._anchor_stats,
                }
            )
        auxiliary = self.gaussian_refine_extra[0]
        if auxiliary.last_stats is not None:
            result.append(
                {
                    "layer": "hr_gaussian_auxiliary",
                    **auxiliary.last_stats,
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
        joint = torch.cat((f_hsi_hr, f_msi), dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)

        anchor_delta = self.gaussian_refine(primitive) - primitive
        auxiliary_delta = self.gaussian_refine_extra[0](primitive) - primitive
        correction, beta, cap_scale = self._bound_auxiliary_correction(
            anchor_delta, auxiliary_delta
        )
        gaussian_delta = anchor_delta + correction

        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            anchor_rms = anchor_delta.detach().square().mean().sqrt()
            auxiliary_rms = auxiliary_delta.detach().square().mean().sqrt()
            correction_rms = correction.detach().square().mean().sqrt()
            self._anchor_stats = {
                "auxiliary_beta": float(beta.detach()),
                "auxiliary_beta_max": self.auxiliary_beta_max,
                "correction_rms_cap_ratio": self.correction_rms_cap_ratio,
                "anchor_delta_rms": float(anchor_rms),
                "auxiliary_delta_rms": float(auxiliary_rms),
                "correction_rms": float(correction_rms),
                "correction_to_anchor_rms_ratio": float(
                    correction_rms / (anchor_rms + 1e-12)
                ),
                "correction_cap_active_ratio": float(
                    (cap_scale.detach() < 0.999999).float().mean()
                ),
            }
        return prediction


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
