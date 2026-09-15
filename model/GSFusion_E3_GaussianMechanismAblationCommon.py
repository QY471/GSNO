"""Strict Gaussian-mechanism ablations on the frozen E3 reconstruction trunk.

Every variant keeps the E3 HSI/MSI encoders, HR fusion, primitive embedding,
decoder, loss, and training protocol unchanged.  Only one Gaussian rendering
mechanism is changed at a time:

* canvas_scaled_std: keep the 64x64 training-patch std exactly equal to E3,
  but scale that std with the current HR canvas at inference;
* raw_sum: return the Gaussian weighted numerator without density division;
* fixed_window: use the historical fixed raster_ratio=0.1 support instead of
  each primitive's adaptive 3-sigma support;
* three_layer: apply three independent current E3 Gaussian residuals in a row.

The canvas-scaled variant intentionally calibrates its scale at 64 pixels.  It
therefore isolates the patch-to-full-image scale coupling without also changing
the E3 std range during training.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    HRAdaptiveGaussianResidual,
    compute_loss,
    sam_loss,
)


class MechanismGaussianResidual(HRAdaptiveGaussianResidual):
    """E3 Gaussian residual with one explicitly selected mechanism change."""

    def __init__(
        self,
        dim: int,
        *,
        canvas_scaled_std: bool = False,
        normalize_density: bool = True,
        adaptive_window: bool = True,
        training_canvas_size: int = 64,
    ) -> None:
        super().__init__(dim=dim)
        self.canvas_scaled_std = bool(canvas_scaled_std)
        self.normalize_density = bool(normalize_density)
        self.adaptive_window = bool(adaptive_window)
        self.training_canvas_size = int(training_canvas_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        raw = self.geometry_head(x)
        raw = raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )

        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(raw[..., 1:2])

        if self.canvas_scaled_std:
            reference_extent = float(max(self.training_canvas_size - 1, 1))
            scale_x = float(max(width - 1, 1)) / reference_extent
            scale_y = float(max(height - 1, 1)) / reference_extent
            std_px = torch.cat(
                [std_scalar_px * scale_x, std_scalar_px * scale_y], dim=-1
            )
        else:
            scale_x = 1.0
            scale_y = 1.0
            std_px = std_scalar_px.expand(-1, -1, 2)

        # Keep center and circular geometry identical to E3.  Offset is tested
        # separately only after the std-unit question has been isolated.
        offset_px = raw.new_zeros(batch, height * width, 2)
        rho = raw.new_zeros(batch, height * width, 1)
        base_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px + offset_px

        delta_value = self.residual_value_head(x)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        ones = delta_value.new_ones(batch, height * width, 1)
        values_with_density = torch.cat([delta_value, ones], dim=-1)

        if self.adaptive_window:
            max_scale = max(scale_x, scale_y)
            raster_ratio = min(
                1.0,
                max(
                    self.sigma_radius * self.std_max_px * max_scale
                    / max(width, 1),
                    self.sigma_radius * self.std_max_px * max_scale
                    / max(height, 1),
                ),
            )
        else:
            raster_ratio = 0.1

        rasterized = self.rasterizer(
            opacity.float(),
            means_px.float(),
            std_px.float(),
            rho.float(),
            values_with_density.float(),
            height,
            width,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=self.adaptive_window,
            sigma_radius=self.sigma_radius,
        )
        rasterized = rasterized.permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels:channels + 1]
        if self.normalize_density:
            gaussian_delta = numerator / density.clamp_min(1e-6)
        else:
            gaussian_delta = numerator
        out = x + gaussian_delta

        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_std_x_mean_px": float(std_px[..., 0].detach().mean()),
                "hrgs_std_y_mean_px": float(std_px[..., 1].detach().mean()),
                "hrgs_std_min_px": float(std_px.detach().min()),
                "hrgs_std_max_px": float(std_px.detach().max()),
                "hrgs_rho_abs_mean": 0.0,
                "hrgs_offset_abs_mean_px": 0.0,
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(delta_value.detach().abs().mean()),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(x_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (x_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": float(self.adaptive_window),
                "hrgs_sigma_radius": float(self.sigma_radius),
                "ablation_canvas_scaled_std": float(self.canvas_scaled_std),
                "ablation_density_normalized": float(self.normalize_density),
                "ablation_std_canvas_scale_x": float(scale_x),
                "ablation_std_canvas_scale_y": float(scale_y),
            }
        return out


class _GaussianMechanismGSFusion(E3GSFusion):
    canvas_scaled_std = False
    normalize_density = True
    adaptive_window = True
    gaussian_layers = 1
    variant_name = "e3_control"

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
        # Construct experiment-only replacements without perturbing the global
        # seeded initialization stream used by the shared E3 trunk.
        rng_state = torch.get_rng_state()
        self.gaussian_refine = MechanismGaussianResidual(
            dim,
            canvas_scaled_std=self.canvas_scaled_std,
            normalize_density=self.normalize_density,
            adaptive_window=self.adaptive_window,
        )
        self.gaussian_refine_extra = nn.ModuleList(
            [
                MechanismGaussianResidual(dim)
                for _ in range(max(int(self.gaussian_layers) - 1, 0))
            ]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            f"Strict E3 Gaussian mechanism ablation: {self.variant_name}; "
            "all non-Gaussian E3 paths are unchanged"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        for layer in getattr(self, "gaussian_refine_extra", []):
            layer.reset_residual_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        layers = [self.gaussian_refine, *self.gaussian_refine_extra]
        result = []
        for index, layer in enumerate(layers, start=1):
            if layer.last_stats is not None:
                result.append({"layer": f"hr_gaussian_{index}", **layer.last_stats})
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
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)
        transported = self.gaussian_refine(primitive)
        for layer in self.gaussian_refine_extra:
            transported = layer(transported)
        gaussian_delta = transported - primitive
        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


class CanvasScaledStdGSFusion(_GaussianMechanismGSFusion):
    canvas_scaled_std = True
    variant_name = "canvas_scaled_std_only"


class RawSumGSFusion(_GaussianMechanismGSFusion):
    normalize_density = False
    variant_name = "raw_scatter_sum_without_density_normalization"


class FixedWindowGSFusion(_GaussianMechanismGSFusion):
    adaptive_window = False
    variant_name = "fixed_raster_ratio_0.1_window"


class ThreeLayerGSFusion(_GaussianMechanismGSFusion):
    gaussian_layers = 3
    variant_name = "three_sequential_gaussian_residual_layers"


__all__ = [
    "CanvasScaledStdGSFusion",
    "RawSumGSFusion",
    "FixedWindowGSFusion",
    "ThreeLayerGSFusion",
    "MechanismGaussianResidual",
    "compute_loss",
    "sam_loss",
]
