"""E3-GDI-v1 with fixed-HR Gaussian difference integral encoders.

Only the two ADCI front-end encoders are replaced. Fusion, primitive
embedding, circular Gaussian refinement, spectral projection, and bicubic
residual reconstruction are inherited unchanged from the formal E3 design.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import compute_loss, sam_loss
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    HRAdaptiveGaussianResidual,
)


def _gradient_l2(parameters) -> float:
    total = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        squared = gradient.detach().float().square().sum()
        total = squared if total is None else total + squared
    return 0.0 if total is None else float(total.sqrt())


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class PointwiseGatedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        eps: float = 1e-6,
        residual_init: float = 1e-3,
    ) -> None:
        super().__init__()
        hidden = int(round(dim * expansion))
        self.norm = LayerNorm2d(dim, eps=eps)
        self.in_proj = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden, dim, kernel_size=1)
        self.res_scale = nn.Parameter(
            torch.full((1, dim, 1, 1), float(residual_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        delta = self.out_proj(F.gelu(value) * torch.sigmoid(gate))
        return x + self.res_scale * delta


class GaussianDifferenceIntegralBlock(nn.Module):
    """Fixed-HR multi-scale Gaussian difference integral."""

    def __init__(
        self,
        dim: int,
        dilations: Sequence[int] = (1, 2, 4),
        sigmas: Sequence[float] = (1.0, 2.0, 4.0),
        eps: float = 1e-6,
        padding_mode: str = "reflect",
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if len(dilations) != len(sigmas):
            raise ValueError("dilations and sigmas must have the same length")
        if not dilations:
            raise ValueError("at least one Gaussian scale is required")
        if any(int(dilation) <= 0 for dilation in dilations):
            raise ValueError("all dilations must be positive")
        if any(float(sigma) <= 0 for sigma in sigmas):
            raise ValueError("all sigmas must be positive")
        if padding_mode not in ("reflect", "replicate"):
            raise ValueError("padding_mode must be reflect or replicate")

        self.dim = int(dim)
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.num_scales = len(self.dilations)
        self.eps = float(eps)
        self.padding_mode = padding_mode

        self.norm = LayerNorm2d(dim, eps=eps)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale_gate = nn.Conv2d(dim, self.num_scales, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.gamma = nn.Parameter(
            torch.full((1, dim, 1, 1), float(gamma_init))
        )

        self.base_offsets: Tuple[Tuple[int, int], ...] = tuple(
            (offset_y, offset_x)
            for offset_y in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
        )
        for scale_index, (dilation, sigma) in enumerate(
            zip(self.dilations, self.sigmas)
        ):
            weights = []
            for offset_y, offset_x in self.base_offsets:
                distance_y = float(offset_y * dilation)
                distance_x = float(offset_x * dilation)
                weights.append(
                    math.exp(
                        -(distance_x ** 2 + distance_y ** 2)
                        / (2.0 * sigma ** 2)
                    )
                )
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(self.eps)
            self.register_buffer(
                f"gaussian_weights_{scale_index}",
                weight_tensor,
                persistent=True,
            )

        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)
        self.last_stats: Optional[Dict[str, float]] = None

    def _gaussian_average(
        self,
        value: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        dilation = self.dilations[scale_index]
        weights = getattr(self, f"gaussian_weights_{scale_index}").to(
            device=value.device,
            dtype=value.dtype,
        )
        _, _, height, width = value.shape
        if self.padding_mode == "reflect" and (
            height <= dilation or width <= dilation
        ):
            raise ValueError(
                "reflect padding requires H and W larger than dilation; "
                f"got {(height, width)} and dilation={dilation}"
            )
        padded = F.pad(
            value,
            (dilation, dilation, dilation, dilation),
            mode=self.padding_mode,
        )
        accumulator = torch.zeros_like(value)
        for offset_index, (offset_y, offset_x) in enumerate(self.base_offsets):
            y_start = dilation + offset_y * dilation
            x_start = dilation + offset_x * dilation
            shifted = padded[
                :,
                :,
                y_start : y_start + height,
                x_start : x_start + width,
            ]
            accumulator = accumulator + weights[offset_index] * shifted
        return accumulator

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        normalized = self.norm(x)
        value = self.value_proj(normalized)
        scale_weights = torch.softmax(self.scale_gate(normalized), dim=1)
        differences = []
        for scale_index in range(self.num_scales):
            smoothed = self._gaussian_average(value, scale_index)
            differences.append(smoothed - value)
        stacked = torch.stack(differences, dim=1)
        mixed = (stacked * scale_weights.unsqueeze(2)).sum(dim=1)
        delta = self.out_proj(mixed)
        output = x + self.gamma * delta

        if not self.training:
            with torch.no_grad():
                detached_weights = scale_weights.detach()
                beta_mean = detached_weights.mean(dim=(0, 2, 3))
                beta_entropy = -(
                    detached_weights
                    * torch.log(detached_weights.clamp_min(1e-8))
                ).sum(dim=1).mean()
                self.last_stats = {
                    "gdi_input_abs_mean": float(x.detach().abs().mean()),
                    "gdi_value_abs_mean": float(value.detach().abs().mean()),
                    "gdi_mixed_abs_mean": float(mixed.detach().abs().mean()),
                    "gdi_delta_abs_mean": float(delta.detach().abs().mean()),
                    "gdi_output_change_abs_mean": float(
                        (output.detach() - x.detach()).abs().mean()
                    ),
                    "gdi_gamma_abs_mean": float(self.gamma.detach().abs().mean()),
                    "gdi_beta_entropy": float(beta_entropy),
                    **{
                        f"gdi_beta_scale_{scale_index}_mean": float(
                            beta_mean[scale_index]
                        )
                        for scale_index in range(self.num_scales)
                    },
                }

        if not return_aux:
            return output
        return output, {
            "value": value,
            "scale_weights": scale_weights,
            "differences": stacked,
            "mixed": mixed,
            "delta": delta,
        }


class GSFusion(nn.Module):
    """Formal E3 decoder with pointwise lifting and fixed-HR GDI encoders."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        pointwise_blocks: int = 2,
        pointwise_expansion: float = 2.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, kernel_size=1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, kernel_size=1)
        self.hsi_point_blocks = nn.ModuleList(
            [
                PointwiseGatedBlock(dim, expansion=pointwise_expansion)
                for _ in range(pointwise_blocks)
            ]
        )
        self.msi_point_blocks = nn.ModuleList(
            [
                PointwiseGatedBlock(dim, expansion=pointwise_expansion)
                for _ in range(pointwise_blocks)
            ]
        )
        self.hsi_gdi = GaussianDifferenceIntegralBlock(dim)
        self.msi_gdi = GaussianDifferenceIntegralBlock(dim)

        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.gaussian_refine = HRAdaptiveGaussianResidual(dim)
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc2 = nn.Conv2d(dim, num_bands, kernel_size=1)
        self.primitive_input = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self._last_gradient_stats: Dict[str, float] = {}
        self._gradient_stat_calls = 0
        self.arch_summary = (
            "E3-GDI-v1: Conv1x1/PointwiseGatedBlockx2 per branch; HSI bicubic "
            "lifting; fixed-HR GDI dilations=(1,2,4), sigmas=(1,2,4); "
            "formal E3 fusion, primitive embedding, circular Gaussian, and head"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        for block in (self.hsi_gdi, self.msi_gdi):
            nn.init.zeros_(block.scale_gate.weight)
            nn.init.zeros_(block.scale_gate.bias)
            nn.init.zeros_(block.gamma)

    def collect_gradient_stats(self) -> Dict[str, float]:
        self._gradient_stat_calls += 1
        if self._gradient_stat_calls > 2 and self._gradient_stat_calls % 250:
            return dict(self._last_gradient_stats)
        self._last_gradient_stats = {
            "grad_hsi_gdi_gamma": _gradient_l2([self.hsi_gdi.gamma]),
            "grad_msi_gdi_gamma": _gradient_l2([self.msi_gdi.gamma]),
            "grad_hsi_scale_gate": _gradient_l2(
                self.hsi_gdi.scale_gate.parameters()
            ),
            "grad_msi_scale_gate": _gradient_l2(
                self.msi_gdi.scale_gate.parameters()
            ),
            "grad_hsi_value_proj": _gradient_l2(
                self.hsi_gdi.value_proj.parameters()
            ),
            "grad_msi_value_proj": _gradient_l2(
                self.msi_gdi.value_proj.parameters()
            ),
        }
        return dict(self._last_gradient_stats)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        gaussian_stats = self.gaussian_refine.last_stats
        if gaussian_stats is not None:
            records.append({"layer": "hr_gaussian", **gaussian_stats})
        for branch, block in (("hsi_gdi", self.hsi_gdi), ("msi_gdi", self.msi_gdi)):
            if block.last_stats is not None:
                stats = {"layer": branch, **block.last_stats}
                stats.update(self._last_gradient_stats)
                records.append(stats)
        return records

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        for block in self.hsi_point_blocks:
            f_hsi = block(f_hsi)
        f_h = F.interpolate(
            f_hsi,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        f_h = self.hsi_gdi(f_h)

        f_m = self.shallow_encoder2(hr_msi)
        for block in self.msi_point_blocks:
            f_m = block(f_m)
        f_m = self.msi_gdi(f_m)

        joint = torch.cat((f_h, f_m), dim=1)
        f_fused = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        primitive_with_delta = self.gaussian_refine(e_g)
        gaussian_delta = primitive_with_delta - e_g
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual
