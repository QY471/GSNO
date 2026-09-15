"""E3 controls with an HSI anchor and an explicit detail transport path.

All variants share the same ADCI encoders and primitive embedding. The HR HSI
latent is the reconstruction anchor. Cross-modal detail can enter the decoder
only through the selected operator; there is no fused-feature bypass.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


def _resolve_adaptive_gaussian_rasterizer():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class _ConditionedDetailOperator(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.condition_head = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_operator_init(self, std_raw_bias: float) -> None:
        nn.init.zeros_(self.condition_head[-1].weight)
        nn.init.zeros_(self.condition_head[-1].bias)
        with torch.no_grad():
            self.condition_head[-1].bias[1] = float(std_raw_bias)

    @staticmethod
    def _condition(detail: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        if detail.shape != msi.shape:
            raise ValueError(
                f"detail shape {tuple(detail.shape)} must match MSI feature "
                f"shape {tuple(msi.shape)}"
            )
        return torch.cat([detail, msi], dim=1)


class GaussianZeroSumDetailOperator(_ConditionedDetailOperator):
    """Density-normalized Gaussian smoothing minus the source detail field."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        initial_std_px: float = 0.55,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__(dim)
        if not std_min_px < initial_std_px < std_max_px:
            raise ValueError("initial_std_px must lie strictly inside std bounds")
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.std_min_px = float(std_min_px)
        self.std_max_px = float(std_max_px)
        self.initial_std_px = float(initial_std_px)
        self.sigma_radius = float(sigma_radius)

    def reset_operator_init(self) -> None:
        initial_unit = (
            (self.initial_std_px - self.std_min_px)
            / (self.std_max_px - self.std_min_px)
        )
        std_raw_bias = math.log(initial_unit / (1.0 - initial_unit))
        super().reset_operator_init(std_raw_bias)

    @staticmethod
    def _pixel_centers(height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).view(1, height * width, 2)

    def forward(self, detail: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = detail.shape
        raw = self.condition_head(self._condition(detail, msi))
        raw = raw.permute(0, 2, 3, 1).contiguous().view(batch, height * width, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(raw[..., 1:2])
        std_px = std_scalar_px.expand(-1, -1, 2)
        rho = raw.new_zeros(batch, height * width, 1)
        means_px = self._pixel_centers(
            height, width, detail.device, detail.dtype
        ).expand(batch, height * width, 2)

        flat_detail = detail.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        payload = torch.cat(
            [flat_detail, flat_detail.new_ones(batch, height * width, 1)], dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_px / max(width, 1),
                self.sigma_radius * self.std_max_px / max(height, 1),
            ),
        )
        rendered = self.rasterizer(
            opacity.float(),
            means_px.float(),
            std_px.float(),
            rho.float(),
            payload.float(),
            height,
            width,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()
        density = rendered[:, channels:channels + 1]
        smoothed = rendered[:, :channels] / density.clamp_min(1e-6)
        transported = smoothed - detail

        with torch.no_grad():
            detail_abs = detail.detach().abs().mean()
            transported_abs = transported.detach().abs().mean()
            self.last_stats = {
                "detail_opacity_mean": float(opacity.detach().mean()),
                "detail_opacity_std": float(opacity.detach().std()),
                "detail_std_mean_px": float(std_scalar_px.detach().mean()),
                "detail_std_min_px": float(std_scalar_px.detach().min()),
                "detail_std_max_px": float(std_scalar_px.detach().max()),
                "detail_std_lower_bound_fraction": float(
                    (std_scalar_px.detach() <= self.std_min_px + 0.01).float().mean()
                ),
                "detail_std_upper_bound_fraction": float(
                    (std_scalar_px.detach() >= self.std_max_px - 0.01).float().mean()
                ),
                "detail_density_min": float(density.detach().min()),
                "detail_density_mean": float(density.detach().mean()),
                "detail_source_abs_mean": float(detail_abs),
                "detail_transport_abs_mean": float(transported_abs),
                "detail_transport_source_ratio": float(
                    transported_abs / (detail_abs + 1e-8)
                ),
                "detail_zero_sum": 1.0,
                "detail_raster_ratio": float(raster_ratio),
            }
        return transported


class PointwiseDetailOperator(_ConditionedDetailOperator):
    """Per-pixel detail injection with no spatial transport."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.reset_operator_init()

    def reset_operator_init(self) -> None:
        initial_unit = (0.55 - 0.30) / (1.50 - 0.30)
        super().reset_operator_init(math.log(initial_unit / (1.0 - initial_unit)))

    def forward(self, detail: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        raw = self.condition_head(self._condition(detail, msi))
        gain = torch.sigmoid(raw[:, 0:1]) * (0.5 + torch.sigmoid(raw[:, 1:2]))
        transported = gain * detail
        with torch.no_grad():
            detail_abs = detail.detach().abs().mean()
            transported_abs = transported.detach().abs().mean()
            self.last_stats = {
                "detail_gain_mean": float(gain.detach().mean()),
                "detail_gain_std": float(gain.detach().std()),
                "detail_source_abs_mean": float(detail_abs),
                "detail_transport_abs_mean": float(transported_abs),
                "detail_transport_source_ratio": float(
                    transported_abs / (detail_abs + 1e-8)
                ),
                "detail_zero_sum": 0.0,
            }
        return transported


class ConvZeroSumDetailOperator(_ConditionedDetailOperator):
    """Learned depthwise 3x3 smoothing minus the source detail field."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.kernel_logits = nn.Parameter(torch.zeros(dim, 1, 3, 3))
        self.reset_operator_init()

    def reset_operator_init(self) -> None:
        initial_unit = (0.55 - 0.30) / (1.50 - 0.30)
        super().reset_operator_init(math.log(initial_unit / (1.0 - initial_unit)))
        nn.init.zeros_(self.kernel_logits)

    def forward(self, detail: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        raw = self.condition_head(self._condition(detail, msi))
        gain = torch.sigmoid(raw[:, 0:1]) * (0.5 + torch.sigmoid(raw[:, 1:2]))
        kernel = torch.softmax(self.kernel_logits.view(self.dim, 1, 9), dim=-1)
        kernel = kernel.view(self.dim, 1, 3, 3)
        padded = F.pad(detail, (1, 1, 1, 1), mode="replicate")
        smoothed = F.conv2d(padded, kernel, padding=0, groups=self.dim)
        transported = gain * (smoothed - detail)
        with torch.no_grad():
            detail_abs = detail.detach().abs().mean()
            transported_abs = transported.detach().abs().mean()
            self.last_stats = {
                "detail_gain_mean": float(gain.detach().mean()),
                "detail_kernel_center_mean": float(kernel[:, :, 1, 1].mean()),
                "detail_source_abs_mean": float(detail_abs),
                "detail_transport_abs_mean": float(transported_abs),
                "detail_transport_source_ratio": float(
                    transported_abs / (detail_abs + 1e-8)
                ),
                "detail_zero_sum": 1.0,
            }
        return transported


class _BaseDetailIntegralGSFusion(nn.Module):
    operator_cls = GaussianZeroSumDetailOperator

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.detail_operator = self.operator_cls(dim)
        self.detail_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.arch_summary = (
            "ADCI encoders; F_H is the stable HSI anchor; concat(F_H,F_M) "
            "forms one shared detail embedding D; no fused reconstruction "
            "bypass; only the selected detail operator can inject MSI-HSI "
            "detail before the shared 1x1 spectral decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        self.detail_operator.reset_operator_init()
        nn.init.zeros_(self.detail_out.weight)
        nn.init.zeros_(self.detail_out.bias)
        with torch.no_grad():
            diagonal = min(self.detail_out.out_channels, self.detail_out.in_channels)
            indices = torch.arange(diagonal)
            self.detail_out.weight[indices, indices, 0, 0] = 0.1

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.detail_operator.last_stats
        return [] if stats is None else [{"layer": "detail_operator", **stats}]

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
        detail_input = torch.cat([f_hsi_hr, f_msi], dim=1)
        detail0 = self.primitive_input(detail_input)
        detail = detail0 + self.primitive_residual(detail0)
        transported = self.detail_operator(detail, f_msi)
        refined = f_hsi_hr + self.detail_out(transported)
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


class GaussianDetailIntegralGSFusion(_BaseDetailIntegralGSFusion):
    operator_cls = GaussianZeroSumDetailOperator


class PointwiseDetailInjectionGSFusion(_BaseDetailIntegralGSFusion):
    operator_cls = PointwiseDetailOperator


class ConvZeroSumDetailIntegralGSFusion(_BaseDetailIntegralGSFusion):
    operator_cls = ConvZeroSumDetailOperator


# Keep the repository's generic frozen-checkpoint evaluator compatible. Formal
# training still selects an explicit class through the Train_Cave registry.
GSFusion = GaussianDetailIntegralGSFusion


__all__ = [
    "GSFusion",
    "GaussianDetailIntegralGSFusion",
    "PointwiseDetailInjectionGSFusion",
    "ConvZeroSumDetailIntegralGSFusion",
    "GaussianZeroSumDetailOperator",
    "compute_loss",
    "sam_loss",
]
