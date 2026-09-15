"""Operator1 with standard point queries and a conservative support window."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.operators.GSFusion_E3_GaussianOperator1 import (
    GSFusion as GaussianOperator1GSFusion,
    GaussianOperatorBlock,
    compute_loss,
    sam_loss,
)


class PointQueryConservativeWindowGaussianOperatorBlock(GaussianOperatorBlock):
    """Evaluate Gaussian values at pixel centres without a learned-sigma cutoff."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__(dim, std_min_px, std_max_px, sigma_radius)
        self.candidate_radius_px = self.sigma_radius * self.std_max_px

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if msi_feature.shape != x.shape:
            raise ValueError(
                f"MSI condition shape {tuple(msi_feature.shape)} must equal "
                f"latent shape {tuple(x.shape)}"
            )

        values = x + self.spectral_transform(x)
        geometry = self.geometry_head(torch.cat([values, msi_feature], dim=1))
        geometry = geometry.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(geometry[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(geometry[..., 1:2])
        std_px = std_scalar_px.expand(-1, -1, 2)
        rho = geometry.new_zeros(batch, height * width, 1)
        means_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)

        flat_values = values.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            [flat_values, flat_values.new_ones(batch, height * width, 1)],
            dim=-1,
        )
        raster_ratio = min(
            1.0,
            max(
                self.candidate_radius_px / max(width, 1),
                self.candidate_radius_px / max(height, 1),
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
            adaptive_window=False,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels:channels + 1]
        out = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            pointwise_abs = (values - x).detach().abs().mean()
            render_change_abs = (out - values).detach().abs().mean()
            sigma = std_scalar_px.detach()
            axial = torch.exp(-0.5 / sigma.square())
            diagonal = torch.exp(-1.0 / sigma.square())
            nominal_total = 1.0 + 4.0 * axial + 4.0 * diagonal
            nominal_offcenter = (nominal_total - 1.0) / nominal_total
            self.last_stats = {
                "gop_opacity_mean": float(opacity.detach().mean()),
                "gop_opacity_std": float(opacity.detach().std()),
                "gop_std_mean_px": float(sigma.mean()),
                "gop_std_min_px": float(sigma.min()),
                "gop_std_max_px": float(sigma.max()),
                "gop_density_min": float(density.detach().min()),
                "gop_density_mean": float(density.detach().mean()),
                "gop_input_abs_mean": float(input_abs),
                "gop_pointwise_update_abs_mean": float(pointwise_abs),
                "gop_render_change_abs_mean": float(render_change_abs),
                "gop_pointwise_input_ratio": float(
                    pointwise_abs / (input_abs + 1e-8)
                ),
                "gop_render_input_ratio": float(
                    render_change_abs / (input_abs + 1e-8)
                ),
                "gop_nominal_offcenter_mass_mean": float(
                    nominal_offcenter.mean()
                ),
                "gop_forced_render": 1.0,
                "gop_point_query": 1.0,
                "gop_adaptive_window": 0.0,
                "gop_conservative_window": 1.0,
                "gop_candidate_radius_px": float(self.candidate_radius_px),
                "gop_candidate_from_sigma_radius": float(self.sigma_radius),
            }
        return out


class GSFusion(GaussianOperator1GSFusion):
    """Operator1 whose renderer preserves standard point-query semantics."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        rng_state = torch.get_rng_state()
        self.operator_blocks = nn.ModuleList(
            [PointQueryConservativeWindowGaussianOperatorBlock(dim)]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "Operator1 topology and standard point-query Gaussian values are "
            "unchanged; a 3*std_max physical-pixel candidate window replaces "
            "the discontinuous per-Gaussian 3*std pruning rule"
        )
        self.reset_custom_init()


__all__ = [
    "GSFusion",
    "PointQueryConservativeWindowGaussianOperatorBlock",
    "compute_loss",
    "sam_loss",
]
