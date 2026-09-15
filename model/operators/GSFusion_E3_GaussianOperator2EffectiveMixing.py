"""Mandatory HR Gaussian operator with non-degenerate spatial mixing.

This revision keeps the no-bypass topology of GaussianOperator1 but removes
its easy identity solution.  MSI contributes to the rendered values, and the
operator-specific physical-pixel sigma range starts at 0.45 px so immediate
HR neighbours receive non-negligible weight.  The decoder can only consume
the density-normalized rendered latent; no fused latent bypass is present.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from model.operators.GSFusion_E3_GaussianOperator1 import (
    GSFusion as GaussianOperator1GSFusion,
    GaussianOperatorBlock,
    compute_loss,
    sam_loss,
)


class EffectiveMixingGaussianOperatorBlock(GaussianOperatorBlock):
    """MSI-conditioned values followed by mandatory effective Gaussian mixing."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.45,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
        initial_std_px: float = 0.55,
    ) -> None:
        if not std_min_px <= initial_std_px <= std_max_px:
            raise ValueError("initial_std_px must lie inside the sigma range")
        super().__init__(
            dim=dim,
            std_min_px=std_min_px,
            std_max_px=std_max_px,
            sigma_radius=sigma_radius,
        )
        self.initial_std_px = float(initial_std_px)
        self.msi_value_adapter = nn.Conv2d(dim, dim, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_operator_init(self) -> None:
        # The pointwise/MSI value updates start at zero, while the renderer is
        # already a real spatial operator at epoch zero.
        nn.init.zeros_(self.spectral_transform[-1].weight)
        nn.init.zeros_(self.spectral_transform[-1].bias)
        nn.init.zeros_(self.msi_value_adapter.weight)
        nn.init.zeros_(self.msi_value_adapter.bias)

        geometry_last = self.geometry_head[-1]
        nn.init.zeros_(geometry_last.weight)
        nn.init.zeros_(geometry_last.bias)
        opacity_fraction = (0.50 - 0.05) / 0.95
        sigma_fraction = (
            (self.initial_std_px - self.std_min_px)
            / (self.std_max_px - self.std_min_px)
        )
        geometry_last.bias.data[0] = math.log(
            opacity_fraction / (1.0 - opacity_fraction)
        )
        geometry_last.bias.data[1] = math.log(
            sigma_fraction / (1.0 - sigma_fraction)
        )

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if msi_feature.shape != x.shape:
            raise ValueError(
                f"MSI condition shape {tuple(msi_feature.shape)} must equal "
                f"latent shape {tuple(x.shape)}"
            )

        spectral_update = self.spectral_transform(x)
        msi_update = self.msi_value_adapter(msi_feature)
        values = x + spectral_update + msi_update
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
                self.sigma_radius * self.std_max_px / max(width, 1),
                self.sigma_radius * self.std_max_px / max(height, 1),
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
        out = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            spectral_abs = spectral_update.detach().abs().mean()
            msi_abs = msi_update.detach().abs().mean()
            render_change_abs = (out - values).detach().abs().mean()
            sigma = std_scalar_px.detach()
            axial = torch.exp(-0.5 / sigma.square())
            diagonal = torch.exp(-1.0 / sigma.square())
            nominal_total = 1.0 + 4.0 * axial + 4.0 * diagonal
            nominal_offcenter = (nominal_total - 1.0) / nominal_total
            lower_bound_fraction = (
                sigma <= self.std_min_px + 1e-4
            ).float().mean()
            self.last_stats = {
                "gop_opacity_mean": float(opacity.detach().mean()),
                "gop_opacity_std": float(opacity.detach().std()),
                "gop_std_mean_px": float(sigma.mean()),
                "gop_std_min_px": float(sigma.min()),
                "gop_std_max_px": float(sigma.max()),
                "gop_std_lower_bound_fraction": float(lower_bound_fraction),
                "gop_nominal_offcenter_mass_mean": float(
                    nominal_offcenter.mean()
                ),
                "gop_density_min": float(density.detach().min()),
                "gop_density_mean": float(density.detach().mean()),
                "gop_input_abs_mean": float(input_abs),
                "gop_spectral_update_abs_mean": float(spectral_abs),
                "gop_msi_value_update_abs_mean": float(msi_abs),
                "gop_render_change_abs_mean": float(render_change_abs),
                "gop_render_input_ratio": float(
                    render_change_abs / (input_abs + 1e-8)
                ),
                "gop_forced_render": 1.0,
                "gop_effective_mixing_revision": 2.0,
                "gop_adaptive_window": 1.0,
                "gop_sigma_radius": float(self.sigma_radius),
            }
        return out


class GSFusion(GaussianOperator1GSFusion):
    """One mandatory, MSI-value-conditioned effective Gaussian operator."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        operator_std_min_px: float = 0.45,
        operator_initial_std_px: float = 0.55,
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
            [
                EffectiveMixingGaussianOperatorBlock(
                    dim=dim,
                    std_min_px=operator_std_min_px,
                    initial_std_px=operator_initial_std_px,
                )
            ]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "ADCI retained; 1x1 alignment has no decoder bypass; MSI enters "
            "the rendered values; one physical-pixel Gaussian renderer with "
            "effective neighbour mixing produces the complete HR latent"
        )
        self.reset_custom_init()


__all__ = [
    "GSFusion",
    "EffectiveMixingGaussianOperatorBlock",
    "compute_loss",
    "sam_loss",
]
