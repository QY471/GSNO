"""Chunk-free pure PyTorch adaptive Gaussian splatting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GaussianRasterizerPyTorch(nn.Module):
    """Density-aware elliptical Gaussian rasterizer using torch scatter_add."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = int(num_channels)

    def forward(
        self,
        opacity: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        rhos: torch.Tensor,
        colors: torch.Tensor,
        image_height: int,
        image_width: int,
        scale_factor: int,
        raster_ratio: float,
        debug: bool = False,
        adaptive_window: bool = False,
        sigma_radius: float = 3.0,
    ) -> torch.Tensor:
        del scale_factor, raster_ratio, debug
        batch, primitive_count, channels = colors.shape
        if channels != self.num_channels:
            raise ValueError(
                f"colors has {channels} channels, expected {self.num_channels}"
            )
        if tuple(opacity.shape) != (batch, primitive_count, 1):
            raise ValueError("opacity must have shape [B,N,1]")
        if tuple(means.shape) != (batch, primitive_count, 2):
            raise ValueError("means must have shape [B,N,2]")
        if tuple(stds.shape) != (batch, primitive_count, 2):
            raise ValueError("stds must have shape [B,N,2]")
        if tuple(rhos.shape) != (batch, primitive_count, 1):
            raise ValueError("rhos must have shape [B,N,1]")

        height = int(image_height)
        width = int(image_width)
        if min(height, width) <= 0:
            raise ValueError("image dimensions must be positive")

        opacity = opacity.float()
        means = means.float()
        stds = stds.float().clamp_min(1e-6)
        rhos = rhos.float().clamp(-0.95, 0.95)
        colors = colors.float()
        std_x = stds[..., 0]
        std_y = stds[..., 1]
        rho = rhos[..., 0]
        rho_denominator = (1.0 - rho.square()).clamp_min(1e-6)
        normalization = (
            2.0
            * math.pi
            * std_x
            * std_y
            * torch.sqrt(rho_denominator)
        ).clamp_min(1e-8)

        if adaptive_window:
            max_radius = int(
                math.ceil(
                    float((sigma_radius * torch.maximum(std_x, std_y)).detach().max())
                )
            ) + 1
        else:
            max_radius = max(height, width)

        center_x = torch.floor(means[..., 0]).to(torch.long)
        center_y = torch.floor(means[..., 1]).to(torch.long)
        flat_size = height * width
        numerator = colors.new_zeros(batch, flat_size, channels)
        for offset_y in range(-max_radius, max_radius + 1):
            target_y = center_y + offset_y
            valid_y = (target_y >= 0) & (target_y < height)
            target_y_clamped = target_y.clamp(0, height - 1)
            delta_y = target_y_clamped.float() - means[..., 1]
            for offset_x in range(-max_radius, max_radius + 1):
                target_x = center_x + offset_x
                valid = valid_y & (target_x >= 0) & (target_x < width)
                target_x_clamped = target_x.clamp(0, width - 1)
                delta_x = target_x_clamped.float() - means[..., 0]

                normalized_x = delta_x / std_x
                normalized_y = delta_y / std_y
                quadratic = (
                    normalized_x.square()
                    - 2.0 * rho * normalized_x * normalized_y
                    + normalized_y.square()
                ) / rho_denominator
                weight = opacity[..., 0] * torch.exp(-0.5 * quadratic)
                weight = weight / normalization
                if adaptive_window:
                    support = (delta_x.abs() <= sigma_radius * std_x) & (
                        delta_y.abs() <= sigma_radius * std_y
                    )
                    valid = valid & support
                weight = weight * valid.to(weight.dtype)

                linear_index = target_y_clamped * width + target_x_clamped
                value_index = linear_index.unsqueeze(-1).expand(-1, -1, channels)
                numerator = numerator.scatter_add(
                    1, value_index, colors * weight.unsqueeze(-1)
                )

        return numerator.view(batch, height, width, self.num_channels)


__all__ = ["GaussianRasterizerPyTorch"]
