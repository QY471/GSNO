"""Mandatory Gaussian operator with pixel-cell-integrated 3-sigma rendering."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

from model.operators.GSFusion_E3_GaussianOperator1 import (
    GSFusion as GaussianOperator1GSFusion,
    GaussianOperatorBlock,
    compute_loss,
    sam_loss,
)


def _resolve_cell_aware_gaussian_rasterizer():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    extension_root = os.environ.get(
        "GSFUSION_CELL_AWARE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "cell_aware3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_cell_aware_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class CellIntegratedRasterizerAdapter(nn.Module):
    """Expose the ordinary rasterizer API over the cell-integrated CUDA kernel."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        GaussianRasterizer = _resolve_cell_aware_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(num_channels)

    def forward(
        self,
        opacity,
        means,
        stds,
        rhos,
        colors,
        image_height,
        image_width,
        scale_factor,
        raster_ratio,
        debug=False,
        adaptive_window=False,
        sigma_radius=3.0,
    ):
        if float(scale_factor) != 1.0:
            raise ValueError("Cell-integrated Operator1 currently requires scale_factor=1")
        if means.shape[1] != image_height * image_width:
            raise ValueError("Cell-integrated Operator1 requires one Gaussian per HR cell")
        keys = colors.new_zeros((colors.shape[0], colors.shape[1], 1))
        gamma = colors.new_zeros((1,))
        return self.rasterizer(
            opacity,
            means,
            stds,
            rhos,
            colors,
            keys,
            gamma,
            image_height,
            image_width,
            scale_factor,
            raster_ratio,
            debug=debug,
            adaptive_window=adaptive_window,
            sigma_radius=sigma_radius,
        )


class CellIntegrated3SigmaGaussianOperatorBlock(GaussianOperatorBlock):
    """Keep the original sigma range while removing the self-only dead zone."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__(dim, std_min_px, std_max_px, sigma_radius)
        self.rasterizer = CellIntegratedRasterizerAdapter(dim + 1)

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        out = super().forward(x, msi_feature)
        if self.last_stats is not None:
            sigma = max(self.last_stats["gop_std_mean_px"], 1e-8)
            support = self.sigma_radius
            center_limit = min(0.5 / sigma, support)
            center_mass_1d = math.erf(center_limit / math.sqrt(2.0))
            total_mass_1d = math.erf(support / math.sqrt(2.0))
            center_fraction_2d = (center_mass_1d / total_mass_1d) ** 2
            self.last_stats.update(
                {
                    "gop_cell_integrated": 1.0,
                    "gop_cell_half_width_px": 0.5,
                    "gop_nominal_offcenter_mass_mean": 1.0 - center_fraction_2d,
                    "gop_identity_dead_zone_removed": 1.0,
                }
            )
        return out


class GSFusion(GaussianOperator1GSFusion):
    """Operator1 with the same parameters and a cell-integrated renderer."""

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
            [CellIntegrated3SigmaGaussianOperatorBlock(dim)]
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "Operator1 topology and 0.30-1.50 px learnable sigma are unchanged; "
            "the complete latent passes through a density-normalized Gaussian "
            "whose strict 3-sigma mass is integrated over HR pixel cells"
        )
        self.reset_custom_init()


__all__ = [
    "GSFusion",
    "CellIntegrated3SigmaGaussianOperatorBlock",
    "compute_loss",
    "sam_loss",
]
