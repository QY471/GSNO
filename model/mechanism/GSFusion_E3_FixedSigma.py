"""Strict Fixed-sigma ablation of the formal circular E3 Gaussian residual."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    HRAdaptiveGaussianResidual,
    compute_loss,
    sam_loss,
)


class FixedSigmaGaussianResidual(HRAdaptiveGaussianResidual):
    """Keep learned opacity/value but use one fixed HR-pixel circular sigma."""

    def __init__(self, dim: int, fixed_sigma_hr: float, sigma_radius: float = 3.0):
        if not (0.0 < float(fixed_sigma_hr)):
            raise ValueError("fixed_sigma_hr must be positive")
        super().__init__(dim=dim, sigma_radius=sigma_radius)
        self.register_buffer(
            "fixed_sigma_hr", torch.tensor(float(fixed_sigma_hr), dtype=torch.float32)
        )
        self._last_gradient_stats: Dict[str, float] = {}
        self._gradient_stat_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        raw_map = self.geometry_head(x)
        raw = raw_map.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_scalar_px = raw.new_full(
            (batch, height * width, 1), float(self.fixed_sigma_hr)
        )
        std_px = std_scalar_px.expand(-1, -1, 2)
        offset_px = raw.new_zeros(batch, height * width, 2)
        rho = raw.new_zeros(batch, height * width, 1)

        means_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)
        delta_value = self.residual_value_head(x)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            [delta_value, delta_value.new_ones(batch, height * width, 1)], dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * float(self.fixed_sigma_hr) / max(width, 1),
                self.sigma_radius * float(self.fixed_sigma_hr) / max(height, 1),
            ),
        )
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
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels:channels + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_opacity_min": float(opacity.detach().min()),
                "hrgs_opacity_max": float(opacity.detach().max()),
                "hrgs_std_x_mean_px": float(self.fixed_sigma_hr),
                "hrgs_std_y_mean_px": float(self.fixed_sigma_hr),
                "hrgs_std_min_px": float(self.fixed_sigma_hr),
                "hrgs_std_max_px": float(self.fixed_sigma_hr),
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_max": float(density.detach().max()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_density_std": float(density.detach().std()),
                "hrgs_value_abs_mean": float(delta_value.detach().abs().mean()),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(x_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (x_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "fixed_sigma_hr_pixel": float(self.fixed_sigma_hr),
            }
        return x + gaussian_delta

    def collect_gradient_stats(self) -> Dict[str, float]:
        self._gradient_stat_calls += 1
        if self._gradient_stat_calls > 2 and self._gradient_stat_calls % 250:
            return dict(self._last_gradient_stats)

        output_weight_grad = self.geometry_head[-1].weight.grad
        output_bias_grad = self.geometry_head[-1].bias.grad

        def norm(value) -> float:
            if value is None:
                return 0.0
            return float(value.detach().float().norm())

        self._last_gradient_stats = {
            "opacity_head_weight_grad_l2": norm(
                None if output_weight_grad is None else output_weight_grad[0]
            ),
            "opacity_head_bias_grad_l2": norm(
                None if output_bias_grad is None else output_bias_grad[0]
            ),
            "unused_std_row_weight_grad_l2": norm(
                None if output_weight_grad is None else output_weight_grad[1]
            ),
            "unused_std_row_bias_grad_l2": norm(
                None if output_bias_grad is None else output_bias_grad[1]
            ),
            "value_head_grad_l2": float(
                sum(
                    norm(parameter.grad) ** 2
                    for parameter in self.residual_value_head.parameters()
                ) ** 0.5
            ),
        }
        return dict(self._last_gradient_stats)


class GSFusion(E3GSFusion):
    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        fixed_sigma_hr: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        rng_state = torch.get_rng_state()
        self.gaussian_refine = FixedSigmaGaussianResidual(
            dim=dim, fixed_sigma_hr=fixed_sigma_hr
        )
        torch.set_rng_state(rng_state)
        self.gaussian_refine.reset_residual_init()
        self.arch_summary = (
            "Strict E3 Fixed-sigma: fixed center, circular Gaussian, learned "
            "opacity/value, primitive embedding, density normalization and "
            "adaptive 3sigma support retained; only physical HR-pixel sigma "
            "is constant and independent of sf"
        )

    def collect_gradient_stats(self) -> Dict[str, float]:
        return self.gaussian_refine.collect_gradient_stats()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.gaussian_refine.last_stats
        if stats is None:
            return []
        return [{
            "layer": "hr_gaussian_fixed_sigma",
            **stats,
            **self.gaussian_refine._last_gradient_stats,
        }]


__all__ = [
    "GSFusion",
    "FixedSigmaGaussianResidual",
    "compute_loss",
    "sam_loss",
]
