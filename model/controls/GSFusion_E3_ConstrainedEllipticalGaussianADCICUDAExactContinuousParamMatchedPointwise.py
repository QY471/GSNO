"""DIM80 parameter-matched pointwise control for the formal continuous GSNO.

The complete ADCI-Exact trunk, fusion path, decoder, bicubic base, and
continuous output interface are inherited from the publication model.  The
primitive/Gaussian residual branch is frozen and bypassed, then replaced at
the same residual-add location by a spatially pointwise
Conv1x1-GELU-Conv1x1 mapping with nearly identical trainable capacity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    GSFusion as FormalGSFusion,
    compute_loss,
    sam_loss,
)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def gaussian_branch_parameter_target(model: FormalGSFusion) -> int:
    return sum(
        parameter_count(module)
        for module in (
            model.primitive_input,
            model.primitive_residual,
            model.gaussian_refine,
        )
    )


def choose_hidden_dim(in_dim: int, out_dim: int, target: int) -> Tuple[int, int]:
    candidates = []
    for hidden in range(1, 4097):
        actual = hidden * (in_dim + out_dim + 1) + out_dim
        candidates.append((abs(actual - target), hidden, actual))
    _, hidden, actual = min(candidates)
    return hidden, actual


class ParamMatchedPointwiseLatent(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(inputs)))


class GSFusion(FormalGSFusion):
    """Formal continuous GSNO with Gaussian transport replaced pointwise."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        pointwise_hidden_dim: int = 0,
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
        self.gaussian_parameter_target = gaussian_branch_parameter_target(self)
        hidden, actual = choose_hidden_dim(2 * dim, dim, self.gaussian_parameter_target)
        if int(pointwise_hidden_dim) > 0:
            hidden = int(pointwise_hidden_dim)
            actual = hidden * (3 * dim + 1) + dim
        self.pointwise_hidden_dim = hidden
        self.pointwise_parameter_count = actual

        for module in (
            self.primitive_input,
            self.primitive_residual,
            self.gaussian_refine,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.pointwise_latent = ParamMatchedPointwiseLatent(2 * dim, dim, hidden)
        self._last_stats: Optional[Dict[str, float]] = None
        self._last_gradient_stats: Dict[str, float] = {}
        self.arch_summary = (
            "DIM80 formal ADCI-Exact continuous pointwise capacity control: "
            "the constrained elliptical Gaussian branch is frozen and bypassed, "
            "then replaced by Conv1x1-GELU-Conv1x1 with matched trainable capacity"
        )

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        if hasattr(self, "pointwise_latent"):
            nn.init.xavier_uniform_(self.pointwise_latent.fc1.weight)
            nn.init.zeros_(self.pointwise_latent.fc1.bias)
            nn.init.zeros_(self.pointwise_latent.fc2.weight)
            nn.init.zeros_(self.pointwise_latent.fc2.bias)

    def collect_gradient_stats(self) -> Dict[str, float]:
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
        f_h = F.interpolate(f_hsi, size=reference_size, mode="bicubic", align_corners=False)
        joint = torch.cat((f_h, f_msi), dim=1)
        fused_reference = self.conv0(joint)
        pointwise_reference = self.pointwise_latent(joint)
        fused = fused_reference if query_size == reference_size else F.interpolate(
            fused_reference, size=query_size, mode="bicubic", align_corners=False
        )
        pointwise_delta = pointwise_reference if query_size == reference_size else F.interpolate(
            pointwise_reference, size=query_size, mode="bicubic", align_corners=False
        )
        refined = fused + pointwise_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            delta_abs = pointwise_delta.detach().abs().mean()
            fused_abs = fused.detach().abs().mean()
            final_abs = residual.detach().abs().mean()
            self._last_stats = {
                "pointwise_residual_abs_mean": float(delta_abs),
                "base_latent_abs_mean": float(fused_abs),
                "pointwise_to_base_ratio": float(delta_abs / (fused_abs + 1e-8)),
                "pointwise_to_final_residual_ratio": float(delta_abs / (final_abs + 1e-8)),
                "target_gaussian_trainable_params": float(self.gaussian_parameter_target),
                "actual_pointwise_trainable_params": float(self.pointwise_parameter_count),
                "parameter_difference": float(
                    self.pointwise_parameter_count - self.gaussian_parameter_target
                ),
                "hidden_dim": float(self.pointwise_hidden_dim),
            }
        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_h,
            "F_M": f_msi,
            "F_fused_reference": fused_reference,
            "F_fused": fused,
            "pointwise_delta_reference": pointwise_reference,
            "pointwise_delta": pointwise_delta,
            "gaussian_branch_bypassed": True,
        }


__all__ = [
    "GSFusion",
    "ParamMatchedPointwiseLatent",
    "choose_hidden_dim",
    "gaussian_branch_parameter_target",
    "compute_loss",
    "sam_loss",
]
