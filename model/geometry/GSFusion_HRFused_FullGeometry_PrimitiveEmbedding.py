"""E3 primitive embedding with full learnable Gaussian geometry.

This is the strict geometry counterpart of
``GSFusion_HRFused_Circular_PrimitiveEmbedding``.  The reconstruction path,
primitive embedding, residual value head, density normalization, and adaptive
3-sigma rasterizer are unchanged.  Only the Gaussian geometry is expanded
from circular ``(opacity, scalar_std)`` to
``(opacity, offset_x, offset_y, std_x, std_y, rho)``.
"""

from __future__ import annotations

import torch

from model.geometry.GSFusion_HRFused_AdaptiveGaussianResidual import (
    HRAdaptiveGaussianResidual as FullGeometryGaussianResidual,
)
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as CircularPrimitiveEmbeddingGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(CircularPrimitiveEmbeddingGSFusion):
    """E3 with full geometry and no other architectural change."""

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

        # Replacing an existing registered module preserves its position in
        # the module order.  Preserve the caller RNG state as well; the formal
        # run uses E3's saved epoch-0 state for every shape-compatible tensor
        # and initializes only this expanded output layer independently.
        rng_state = torch.get_rng_state()
        self.gaussian_refine = FullGeometryGaussianResidual(dim)
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E3 strict full-geometry counterpart: identical ADCI backbone, "
            "fusion, primitive embedding, Gaussian value head, normalized "
            "adaptive-3sigma scatter, and decoder; geometry learns opacity, "
            "offset_x/y, std_x/y, and rho"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
