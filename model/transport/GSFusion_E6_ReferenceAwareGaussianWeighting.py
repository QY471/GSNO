"""E6-RAG: E6 with reference-aware Gaussian rasterization weights.

The E6 backbone, primitive embedding, MSI-guided HSI routing, circular
geometry, value head, decoder, and loss remain unchanged.  The only new term
inside rasterization is exp(gamma * cosine(k_p, k_i)), where one shared 1x1
projection maps F_M to normalized 8-D keys at both target and primitive
positions.  gamma is initialized to zero, so the initial forward function is
exactly E6.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveValueCommon import (
    HRAdaptiveGaussianResidualDualSource,
    HRFusedPrimitiveValueBase,
    MSIGuidedHSILocalRouting,
    compute_loss,
    sam_loss,
)


def _resolve_reference_aware_rasterizer():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_root = os.environ.get(
        "GSFUSION_REFERENCE_AWARE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "reference_aware_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_reference_aware_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class ReferenceAwareGaussianResidual(HRAdaptiveGaussianResidualDualSource):
    """Circular adaptive-3sigma renderer with an MSI key similarity term."""

    def __init__(self, dim: int, key_dim: int = 8, **kwargs: object) -> None:
        super().__init__(dim, **kwargs)
        GaussianRasterizer = _resolve_reference_aware_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.key_dim = int(key_dim)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        normalized_keys: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        if transport_x.shape != value_x.shape:
            raise ValueError("transport_x and value_x must have identical BCHW shapes")
        batch, channels, height, width = transport_x.shape
        if normalized_keys.shape != (batch, self.key_dim, height, width):
            raise ValueError(
                "normalized_keys must be B,key_dim,H,W, got "
                f"{tuple(normalized_keys.shape)}"
            )
        if gamma.numel() != 1:
            raise ValueError("gamma must contain exactly one learnable scalar")

        raw = self.geometry_head(transport_x)
        raw = raw.permute(0, 2, 3, 1).contiguous().view(batch, height * width, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(raw[..., 1:2])
        std_px = std_scalar_px.expand(-1, -1, 2)
        offset_px = raw.new_zeros(batch, height * width, 2)
        rho = raw.new_zeros(batch, height * width, 1)

        means_px = self._pixel_centers(
            height, width, transport_x.device, transport_x.dtype
        ).expand(batch, height * width, 2)
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

        delta_value = self.residual_value_head(value_x)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            [delta_value, delta_value.new_ones(batch, height * width, 1)], dim=-1
        )
        keys = normalized_keys.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, self.key_dim
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
            keys.float(),
            gamma.float(),
            height,
            width,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        )
        rasterized = rasterized.permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels : channels + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            transport_abs = transport_x.detach().abs().mean()
            value_abs = delta_value.detach().abs().mean()
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
                "hrgs_offset_max_abs_px": 0.0,
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(value_abs),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(transport_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (transport_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "rag_gamma": float(gamma.detach()),
                "rag_key_dim": float(self.key_dim),
            }
        return gaussian_delta


class GSFusion(HRFusedPrimitiveValueBase):
    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        key_dim: int = 8,
        **_: object,
    ) -> None:
        super().__init__(dim, num_bands, num_msi, adci_layers)
        # Replace only the rasterizer implementation; all E6 Gaussian heads
        # retain their names, shapes, registration order, and initialization.
        old_refine = self.gaussian_refine
        replacement_rng_state = torch.get_rng_state()
        self.gaussian_refine = ReferenceAwareGaussianResidual(dim, key_dim=key_dim)
        self.gaussian_refine.load_state_dict(old_refine.state_dict(), strict=True)
        torch.set_rng_state(replacement_rng_state)

        extra_rng_state = torch.get_rng_state()
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)
        self.reference_key = nn.Conv2d(dim, key_dim, kernel_size=1, bias=False)
        torch.set_rng_state(extra_rng_state)
        self.gamma_raw = nn.Parameter(torch.zeros(1))
        self.key_dim = int(key_dim)
        self.arch_summary = (
            "E6-RAG: frozen E6 architecture plus shared 1x1 MSI keys and "
            "exp(gamma*cosine) inside circular adaptive-3sigma normalized CUDA splatting"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        self.reset_common_init()
        self.routing.reset_output_init()
        nn.init.zeros_(self.gamma_raw)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        base, f_h, f_m, _joint, f_fused, e_g = self._encode_common(lr_hsi, hr_msi)
        routing_delta = self.routing(f_h, f_m)
        value_source = e_g + routing_delta
        normalized_keys = F.normalize(self.reference_key(f_m), p=2, dim=1, eps=1e-6)
        gaussian_delta = self.gaussian_refine(
            transport_x=e_g,
            value_x=value_source,
            normalized_keys=normalized_keys,
            gamma=self.gamma_raw,
        )
        self._record_primitive_stats(
            e_g,
            value_source,
            routing_delta=routing_delta,
            routing=self.routing,
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
