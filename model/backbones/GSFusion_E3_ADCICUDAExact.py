"""Formal circular E3 with checkpoint-compatible ADCI CUDA acceleration.

The architecture, score path, and parameter tensors are identical to formal
E3.  Only the 3x3 neighbor softmax/value reduction uses a Triton-compiled CUDA
kernel without materializing ``v_unfold``.  This module is an engineering
acceleration, not a new learning algorithm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ADCI_Exact import ADCIExact
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as CircularE3,
    compute_loss,
    sam_loss,
)


class GSFusion(CircularE3):
    """Circular E3 whose six ADCI blocks use the exact CUDA execution path."""

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
            "Formal circular E3 with unchanged score equations, checkpoint-"
            "compatible ADCI Exact CUDA/Triton value aggregation; Gaussian path is "
            "unchanged"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
