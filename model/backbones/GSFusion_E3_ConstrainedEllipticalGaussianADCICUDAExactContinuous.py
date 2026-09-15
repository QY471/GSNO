"""Best single-layer E3 with ADCI Exact and continuous Gaussian queries.

The DIM80 constrained-elliptical CUDA-ADCI model is preserved exactly at the
native HR-MSI resolution.  This variant only expresses its fixed-center
Gaussian field in HR-MSI reference coordinates so the same learned field can
be queried at an arbitrary output size.  It adds no gate, routing, factorized
content branch, or trainable parameter.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from model.GSFusion_E3_ConstrainedEllipticalGaussian import (
    ConstrainedEllipticalGaussianResidual,
)
from model.GSFusion_GSNO import compute_loss, sam_loss
from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExact import (
    GSFusion as CUDAExactConstrainedEllipticalGSFusion,
)


class ContinuousConstrainedEllipticalGaussianResidual(
    ConstrainedEllipticalGaussianResidual
):
    """Render the same reference-coordinate elliptical field on any grid."""

    def forward(
        self,
        x: torch.Tensor,
        reference_size: Tuple[int, int],
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        batch, channels, height, width = x.shape
        reference_h, reference_w = map(int, reference_size)
        output_h, output_w = map(int, out_size)
        if (height, width) != (reference_h, reference_w):
            raise ValueError(
                f"primitive grid {(height, width)} must equal reference grid "
                f"{(reference_h, reference_w)}"
            )
        if min(reference_h, reference_w, output_h, output_w) <= 0:
            raise ValueError(
                f"reference_size and out_size must be positive, got "
                f"{reference_size} and {out_size}"
            )

        circular_raw = self.geometry_head(x)
        circular_raw = circular_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(circular_raw[..., 0:1])

        shape_raw = self.anisotropy_head(x)
        shape_raw = shape_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        log_stretch = self.max_log_stretch * torch.tanh(shape_raw[..., 0:1])
        theta = 0.5 * math.pi * torch.tanh(shape_raw[..., 1:2])
        stretch_abs = log_stretch.abs()
        base_min = self.std_min_px * torch.exp(stretch_abs)
        base_max = self.std_max_px * torch.exp(-stretch_abs)
        base_sigma = base_min + (base_max - base_min) * torch.sigmoid(
            circular_raw[..., 1:2]
        )
        sigma_1 = base_sigma * torch.exp(log_stretch)
        sigma_2 = base_sigma * torch.exp(-log_stretch)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        var_1 = sigma_1.square()
        var_2 = sigma_2.square()
        var_x_reference = var_1 * cos_theta.square() + var_2 * sin_theta.square()
        var_y_reference = var_1 * sin_theta.square() + var_2 * cos_theta.square()
        cov_xy_reference = (var_1 - var_2) * sin_theta * cos_theta
        std_x_reference = torch.sqrt(var_x_reference)
        std_y_reference = torch.sqrt(var_y_reference)
        rho = cov_xy_reference / (
            std_x_reference * std_y_reference
        ).clamp_min(1e-12)
        rho = rho.clamp(-0.95, 0.95)

        scale_x = output_w / reference_w
        scale_y = output_h / reference_h
        std_output = torch.cat(
            (std_x_reference * scale_x, std_y_reference * scale_y), dim=-1
        )
        means_reference = self._pixel_centers(
            reference_h, reference_w, x.device, x.dtype
        ).expand(batch, height * width, 2)
        means_output = torch.stack(
            (
                (means_reference[..., 0] + 0.5) * scale_x - 0.5,
                (means_reference[..., 1] + 0.5) * scale_y - 0.5,
            ),
            dim=-1,
        )

        delta_value = self.residual_value_head(x)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            (delta_value, delta_value.new_ones(batch, height * width, 1)), dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_px / max(reference_w, 1),
                self.sigma_radius * self.std_max_px / max(reference_h, 1),
            ),
        )
        rasterized = self.rasterizer(
            opacity.float(),
            means_output.float(),
            std_output.float(),
            rho.float(),
            values_with_density.float(),
            output_h,
            output_w,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels : channels + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            axis_major = torch.maximum(sigma_1, sigma_2)
            axis_minor = torch.minimum(sigma_1, sigma_2)
            axis_ratio = axis_major / axis_minor.clamp_min(1e-12)
            x_abs = x.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_base_sigma_mean_px": float(base_sigma.detach().mean()),
                "hrgs_axis_ratio_mean": float(axis_ratio.detach().mean()),
                "hrgs_axis_ratio_max": float(axis_ratio.detach().max()),
                "hrgs_theta_abs_mean_deg": float(
                    theta.detach().abs().mean() * (180.0 / math.pi)
                ),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": 0.0,
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(delta_value.detach().abs().mean()),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(x_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (x_abs + 1e-8)),
                "hrgs_reference_height": float(reference_h),
                "hrgs_reference_width": float(reference_w),
                "hrgs_output_height": float(output_h),
                "hrgs_output_width": float(output_w),
                "hrgs_output_scale_x": float(scale_x),
                "hrgs_output_scale_y": float(scale_y),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "hrgs_max_axis_ratio_limit": float(self.max_axis_ratio),
            }
        if not return_aux:
            return gaussian_delta
        return gaussian_delta, {
            "density": density,
            "means_reference": means_reference,
            "means_output": means_output,
            "std_output": std_output,
            "rho": rho,
            "opacity": opacity,
            "value": delta_value,
        }


class GSFusion(CUDAExactConstrainedEllipticalGSFusion):
    """DIM80 CUDA-ADCI ellipse with native-equivalent continuous output."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
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
        rng_state = torch.get_rng_state()
        self.gaussian_refine = ContinuousConstrainedEllipticalGaussianResidual(
            dim=dim, max_axis_ratio=max_axis_ratio
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "E3 DIM80 single constrained-elliptical Gaussian; six ADCI Exact "
            "CUDA/Triton blocks; no gate or factorized branch; fixed-center "
            "density-normalized adaptive-3sigma field in native HR-MSI reference "
            "coordinates with arbitrary output_size"
        )
        self.reset_custom_init()

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(int(v) for v in hr_msi.shape[-2:])
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
        f_fused_reference = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        gaussian_delta, gaussian_aux = self.gaussian_refine(
            e_g, reference_size, query_size, return_aux=True
        )
        f_fused = f_fused_reference if query_size == reference_size else F.interpolate(
            f_fused_reference, size=query_size, mode="bicubic", align_corners=False
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_h,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": e0,
            "E_g": e_g,
            "gaussian_delta": gaussian_delta,
            **gaussian_aux,
        }


__all__ = [
    "GSFusion",
    "ContinuousConstrainedEllipticalGaussianResidual",
    "compute_loss",
    "sam_loss",
]
