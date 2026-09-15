"""ADCI Exact LR-cell Gaussian transport with local value context."""

from __future__ import annotations

import torch

from model.backbones.GSFusion_E6_LRPrimitiveGaussianTransportADCICUDAExact import (
    GSFusion as CircularLRPrimitiveGSFusion,
    compute_loss,
    sam_loss,
)
from model.important_model_support.GSFusion_LRPrimitiveValueContextTransport import (
    LRCellValueContextGaussianTransport,
)


class GSFusion(CircularLRPrimitiveGSFusion):
    """Replace only Gaussian value generation with zero-init LR context."""

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
        self.gaussian_transport = LRCellValueContextGaussianTransport(dim=dim)
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "E6 LR-cell transport with zero-initialized depthwise 3x3 value "
            "context, fixed centers, adaptive 3-sigma, density normalization, "
            "and ADCI Exact"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
