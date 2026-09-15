"""LR-cell Gaussian transport with checkpoint-compatible ADCI Exact blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ADCI_Exact import ADCIExact
from model.transport.GSFusion_E6_LRPrimitiveGaussianTransport import (
    GSFusion as LRPrimitiveGaussianTransportGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(LRPrimitiveGaussianTransportGSFusion):
    """Preserve LR-cell transport while accelerating all six ADCI blocks."""

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
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E6 LR-cell Gaussian primitive transport with checkpoint-compatible "
            "ADCI Exact CUDA/Triton aggregation in both three-layer encoders; "
            "MSI footprint, LR routing, Gaussian transport, HR reconstruction, "
            "and decoder are unchanged"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
