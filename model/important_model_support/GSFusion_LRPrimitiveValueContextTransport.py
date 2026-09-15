"""Zero-initialized LR-neighborhood context for Gaussian transport values."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    LRCellGaussianTransportDualSource,
)


class LRCellValueContextGaussianTransport(LRCellGaussianTransportDualSource):
    """Add only a depthwise 3x3 residual before Gaussian value projection."""

    def __init__(
        self,
        dim: int,
        std_min_cell: float = 0.125,
        std_max_cell: float = 1.0,
        std_init_cell: float = 0.25,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
        std_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            dim=dim,
            std_min_cell=std_min_cell,
            std_max_cell=std_max_cell,
            std_init_cell=std_init_cell,
            sigma_radius=sigma_radius,
            density_eps=density_eps,
            std_multiplier=std_multiplier,
        )
        self.value_context = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        nn.init.zeros_(self.value_context.weight)
        nn.init.zeros_(self.value_context.bias)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        contextual_value = value_x + self.value_context(value_x)
        return super().forward(
            transport_x,
            contextual_value,
            out_size,
            return_aux=return_aux,
        )


__all__ = ["LRCellValueContextGaussianTransport"]
