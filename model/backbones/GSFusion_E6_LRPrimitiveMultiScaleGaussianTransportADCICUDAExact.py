"""LR-cell multi-scale Gaussian transport with ADCI Exact CUDA blocks."""

from __future__ import annotations

import torch

from model.backbones.GSFusion_E6_LRPrimitiveGaussianTransportADCICUDAExact import (
    GSFusion as LRPrimitiveCUDAExactGSFusion,
    compute_loss,
    sam_loss,
)
from model.important_model_support.GSFusion_LRPrimitiveMultiScaleTransport import (
    MultiScaleLRCellGaussianTransport,
)


class GSFusion(LRPrimitiveCUDAExactGSFusion):
    """Use three content-weighted support scales for every LR primitive."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        std_multiplier: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            std_multiplier=std_multiplier,
            **kwargs,
        )
        rng_state = torch.get_rng_state()
        self.gaussian_transport = MultiScaleLRCellGaussianTransport(
            dim=dim,
            std_multiplier=std_multiplier,
            expert_scales=(0.5, 1.0, 1.5),
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "E6 LR-cell transport with three MSI-conditioned Gaussian support "
            "experts per cell (0.5x/1.0x/1.5x), density-normalized adaptive "
            "3-sigma rasterization, and ADCI Exact CUDA encoders"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
