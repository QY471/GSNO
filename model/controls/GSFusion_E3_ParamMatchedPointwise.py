"""Strict E3 capacity control with no Gaussian or spatial neighbourhood reads.

The complete formal E3 trunk is retained.  The entire primitive-embedding and
Gaussian correction branch is bypassed and replaced at the same branch input
and residual-add location by a two-layer 1x1 MLP.  Its hidden width is chosen
to match the trainable parameter delta between formal E3 and E3-NoGaussian.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    compute_loss,
    sam_loss,
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def gaussian_branch_parameter_target(model: E3GSFusion) -> int:
    return sum(
        _parameter_count(module)
        for module in (
            model.primitive_input,
            model.primitive_residual,
            model.gaussian_refine,
        )
    )


def choose_hidden_dim(c_in: int, c_out: int, target: int) -> tuple[int, int]:
    candidates = []
    for hidden in range(1, 2049):
        actual = hidden * (c_in + c_out + 1) + c_out
        candidates.append((abs(actual - target), hidden, actual))
    _, hidden, actual = min(candidates)
    return hidden, actual


class ParamMatchedPointwiseLatent(nn.Module):
    """Two 1x1 projections with GELU; no spatial or coordinate operation."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GSFusion(E3GSFusion):
    """Formal E3 with only its Gaussian correction replaced pointwise."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        pointwise_hidden_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        self.gaussian_parameter_target = gaussian_branch_parameter_target(self)
        selected_hidden, actual = choose_hidden_dim(
            2 * dim, dim, self.gaussian_parameter_target
        )
        if int(pointwise_hidden_dim) > 0:
            selected_hidden = int(pointwise_hidden_dim)
            actual = selected_hidden * (3 * dim + 1) + dim
        self.pointwise_hidden_dim = selected_hidden
        self.pointwise_parameter_count = actual

        for module in (
            self.primitive_input,
            self.primitive_residual,
            self.gaussian_refine,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.pointwise_latent = ParamMatchedPointwiseLatent(
            2 * dim, dim, selected_hidden
        )
        self._last_stats: Optional[Dict[str, float]] = None
        self._last_gradient_stats: Dict[str, float] = {}
        self._gradient_stat_calls = 0
        self.arch_summary = (
            "Strict E3 Param-Matched Pointwise: formal ADCI trunk/fusion/head; "
            "Gaussian branch replaced by Conv1x1-GELU-Conv1x1 at the same "
            "joint-input and fused-latent residual-add positions; no Gaussian, "
            "geometry, coordinates, neighbourhood reads, or scale conditions"
        )

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        if hasattr(self, "pointwise_latent"):
            nn.init.zeros_(self.pointwise_latent.fc2.weight)
            nn.init.zeros_(self.pointwise_latent.fc2.bias)

    def collect_gradient_stats(self) -> Dict[str, float]:
        self._gradient_stat_calls += 1
        if self._gradient_stat_calls > 2 and self._gradient_stat_calls % 250:
            return dict(self._last_gradient_stats)

        def grad_l2(parameter: torch.Tensor) -> float:
            if parameter.grad is None:
                return 0.0
            return float(parameter.grad.detach().float().norm())

        self._last_gradient_stats = {
            "pointwise_fc1_grad_l2": grad_l2(self.pointwise_latent.fc1.weight),
            "pointwise_fc2_grad_l2": grad_l2(self.pointwise_latent.fc2.weight),
        }
        return dict(self._last_gradient_stats)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        if self._last_stats is None:
            return []
        return [{"layer": "pointwise_latent", **self._last_stats, **self._last_gradient_stats}]

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
        pointwise_delta = self.pointwise_latent(joint)
        refined = fused + pointwise_delta
        final_residual = self.fc2(F.gelu(self.fc1(refined)))

        with torch.no_grad():
            delta_abs = pointwise_delta.detach().abs().mean()
            fused_abs = fused.detach().abs().mean()
            final_abs = final_residual.detach().abs().mean()
            self._last_stats = {
                "pointwise_residual_abs_mean": float(delta_abs),
                "base_latent_abs_mean": float(fused_abs),
                "pointwise_to_base_ratio": float(delta_abs / (fused_abs + 1e-8)),
                "pointwise_to_final_residual_ratio": float(
                    delta_abs / (final_abs + 1e-8)
                ),
                "target_added_params": float(self.gaussian_parameter_target),
                "actual_added_params": float(self.pointwise_parameter_count),
                "hidden_dim": float(self.pointwise_hidden_dim),
            }
        return base + final_residual


__all__ = [
    "GSFusion",
    "ParamMatchedPointwiseLatent",
    "choose_hidden_dim",
    "gaussian_branch_parameter_target",
    "compute_loss",
    "sam_loss",
]
