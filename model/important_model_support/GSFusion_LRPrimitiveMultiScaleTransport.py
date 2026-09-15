"""Content-conditioned multi-scale Gaussian transport on LR cells."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    LRCellGaussianTransportDualSource,
)


class MultiScaleLRCellGaussianTransport(LRCellGaussianTransportDualSource):
    """Render small, medium, and large Gaussian experts per LR cell."""

    def __init__(
        self,
        dim: int,
        expert_scales: Sequence[float] = (0.5, 1.0, 1.5),
        gate_temperature: float = 1.0,
        expert_specific_values: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(dim=dim, **kwargs)
        scales = tuple(float(scale) for scale in expert_scales)
        if len(scales) < 2 or any(scale <= 0 for scale in scales):
            raise ValueError("expert_scales must contain at least two positive values")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.register_buffer(
            "expert_scales",
            torch.tensor(scales, dtype=torch.float32),
            persistent=True,
        )
        self.gate_temperature = float(gate_temperature)
        self.expert_specific_values = bool(expert_specific_values)
        hidden_dim = max(dim // 4, 8)
        self.expert_logit_head = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, len(scales), 1),
        )
        self.expert_value_residual_head = None
        if self.expert_specific_values:
            self.expert_value_residual_head = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, len(scales) * dim, 1),
            )

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        nn.init.zeros_(self.expert_logit_head[-1].weight)
        nn.init.zeros_(self.expert_logit_head[-1].bias)
        middle = len(self.expert_scales) // 2
        with torch.no_grad():
            self.expert_logit_head[-1].bias.fill_(-2.0)
            self.expert_logit_head[-1].bias[middle] = 2.0
        if self.expert_value_residual_head is not None:
            nn.init.zeros_(self.expert_value_residual_head[-1].weight)
            nn.init.zeros_(self.expert_value_residual_head[-1].bias)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        if transport_x.shape != value_x.shape:
            raise ValueError(
                "transport_x and value_x must have identical BCHW shapes, got "
                f"{tuple(transport_x.shape)} and {tuple(value_x.shape)}"
            )
        batch, channels, h, w = transport_x.shape
        height, width = int(out_size[0]), int(out_size[1])
        if height < h or width < w:
            raise ValueError("target HR grid must not be smaller than the LR grid")

        raw = self.geometry_head(transport_x).permute(0, 2, 3, 1).contiguous()
        raw = raw.view(batch, h * w, 2)
        base_opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        base_std_cell = self.std_min_cell + (
            self.std_max_cell - self.std_min_cell
        ) * torch.sigmoid(raw[..., 1:2])

        expert_logits = self.expert_logit_head(transport_x)
        expert_logits = expert_logits.permute(0, 2, 3, 1).reshape(
            batch, h * w, len(self.expert_scales)
        )
        expert_weights = torch.softmax(
            expert_logits / self.gate_temperature, dim=-1
        )
        scales = self.expert_scales.to(
            device=transport_x.device, dtype=transport_x.dtype
        ).view(1, 1, -1, 1)
        std_cell = base_std_cell.unsqueeze(2) * scales
        scale_x, scale_y = width / w, height / h
        std_x_hr = std_cell * scale_x * self.std_multiplier
        std_y_hr = std_cell * scale_y * self.std_multiplier
        std_hr = torch.cat((std_x_hr, std_y_hr), dim=-1)

        means_hr = self.lr_centers_in_hr(
            h, w, height, width, transport_x.device, transport_x.dtype
        ).expand(batch, -1, -1)
        means_hr = means_hr.unsqueeze(2).expand(-1, -1, len(self.expert_scales), -1)
        opacity = base_opacity.unsqueeze(2) * expert_weights.unsqueeze(-1)

        value = self.residual_value_head(value_x)
        value = value.permute(0, 2, 3, 1).reshape(batch, h * w, channels)
        expert_value = value.unsqueeze(2).expand(
            -1, -1, len(self.expert_scales), -1
        )
        expert_value_residual = None
        if self.expert_value_residual_head is not None:
            expert_value_residual = self.expert_value_residual_head(value_x)
            expert_value_residual = expert_value_residual.view(
                batch, len(self.expert_scales), channels, h, w
            ).permute(0, 3, 4, 1, 2).reshape(
                batch, h * w, len(self.expert_scales), channels
            )
            expert_value = expert_value + expert_value_residual
        values_with_density = torch.cat(
            (expert_value, expert_value.new_ones(*expert_value.shape[:-1], 1)),
            dim=-1,
        )

        primitive_count = h * w * len(self.expert_scales)
        means_hr = means_hr.reshape(batch, primitive_count, 2)
        std_hr = std_hr.reshape(batch, primitive_count, 2)
        opacity = opacity.reshape(batch, primitive_count, 1)
        values_with_density = values_with_density.reshape(
            batch, primitive_count, channels + 1
        )
        offset_hr = means_hr.new_zeros(batch, primitive_count, 2)
        rho = means_hr.new_zeros(batch, primitive_count, 1)

        max_expert_scale = float(self.expert_scales.max())
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius
                * self.std_max_cell
                * self.std_multiplier
                * max_expert_scale
                / max(w, 1),
                self.sigma_radius
                * self.std_max_cell
                * self.std_multiplier
                * max_expert_scale
                / max(h, 1),
            ),
        )
        rasterized = self.rasterizer(
            opacity.float(),
            means_hr.float(),
            std_hr.float(),
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
        density = rasterized[:, channels : channels + 1]
        gaussian_delta_hr = numerator / density.clamp_min(self.density_eps)

        with torch.no_grad():
            low_density = density < 1e-4
            expert_means = expert_weights.detach().mean(dim=(0, 1))
            self.last_stats = {
                "primitive_count": float(primitive_count),
                "experts_per_cell": float(len(self.expert_scales)),
                "gate_temperature": self.gate_temperature,
                "expert_specific_values": float(self.expert_specific_values),
                "std_multiplier": self.std_multiplier,
                "base_std_cell_mean": float(base_std_cell.detach().mean()),
                "std_cell_mean": float(std_cell.detach().mean()),
                "std_cell_min": float(std_cell.detach().min()),
                "std_cell_max": float(std_cell.detach().max()),
                "std_x_hr_mean": float(std_x_hr.detach().mean()),
                "std_y_hr_mean": float(std_y_hr.detach().mean()),
                "opacity_mean": float(opacity.detach().mean()),
                "density_min": float(density.detach().min()),
                "density_mean": float(density.detach().mean()),
                "density_max": float(density.detach().max()),
                "density_lt_1e_4_ratio": float(low_density.float().mean()),
                "value_abs_mean": float(value.detach().abs().mean()),
                "gaussian_delta_abs_mean": float(
                    gaussian_delta_hr.detach().abs().mean()
                ),
                "adaptive_window": 1.0,
                "sigma_radius": self.sigma_radius,
                "raster_ratio": float(raster_ratio),
            }
            for index, weight in enumerate(expert_means):
                self.last_stats[f"expert_{index}_weight_mean"] = float(weight)
                self.last_stats[f"expert_{index}_scale"] = float(
                    self.expert_scales[index]
                )
            self.last_aux = {
                "opacity": opacity.detach(),
                "base_std_cell": base_std_cell.detach(),
                "std_cell": std_cell.detach(),
                "std_hr": std_hr.detach(),
                "means_hr": means_hr.detach(),
                "offset_hr": offset_hr.detach(),
                "rho": rho.detach(),
                "density": density.detach(),
                "expert_weights": expert_weights.detach(),
            }
            if expert_value_residual is not None:
                self.last_stats["expert_value_residual_abs_mean"] = float(
                    expert_value_residual.detach().abs().mean()
                )
                self.last_aux["expert_value_residual"] = (
                    expert_value_residual.detach()
                )

        if not return_aux:
            return gaussian_delta_hr
        aux: Dict[str, torch.Tensor] = {
            "opacity": opacity,
            "base_std_cell": base_std_cell,
            "std_cell": std_cell,
            "std_hr": std_hr,
            "means_hr": means_hr,
            "offset_hr": offset_hr,
            "rho": rho,
            "density": density,
            "expert_weights": expert_weights,
        }
        if expert_value_residual is not None:
            aux["expert_value_residual"] = expert_value_residual
        return gaussian_delta_hr, aux


__all__ = ["MultiScaleLRCellGaussianTransport"]
