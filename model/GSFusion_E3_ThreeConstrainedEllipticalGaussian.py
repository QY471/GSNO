"""DIM-configurable E3 with three constrained elliptical Gaussian layers.

This combines two already isolated E3 changes without altering the ADCI
encoders, fusion trunk, primitive embedding, decoder, or raw bicubic base:

* three sequential density-normalized adaptive-3sigma Gaussian updates; and
* bounded, area-preserving anisotropy with a maximum principal-axis ratio.

The anisotropy heads are initialized to zero, so every layer is exactly
circular at epoch 0.  A DIM80 circular three-layer run is therefore the paired
control for deciding whether learned anisotropy adds value beyond capacity and
Gaussian depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.GSFusion_E3_ConstrainedEllipticalGaussian import (
    ConstrainedEllipticalGaussianResidual,
)
from model.GSFusion_E3_GaussianMechanismAblationCommon import (
    _GaussianMechanismGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(_GaussianMechanismGSFusion):
    """Three sequential constrained-elliptical E3 Gaussian updates."""

    gaussian_layers = 3
    variant_name = "three_constrained_elliptical_gaussian_layers"

    def __init__(
        self,
        dim: int = 64,
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
            **kwargs,
        )

        # The paired checkpoint supplies every tensor shared with circular E3.
        # Preserve the caller RNG while replacing the temporary circular
        # Gaussian layers by their constrained-elliptical counterparts.
        rng_state = torch.get_rng_state()
        self.gaussian_refine = ConstrainedEllipticalGaussianResidual(
            dim=dim,
            max_axis_ratio=max_axis_ratio,
        )
        self.gaussian_refine_extra = nn.ModuleList(
            [
                ConstrainedEllipticalGaussianResidual(
                    dim=dim,
                    max_axis_ratio=max_axis_ratio,
                )
                for _ in range(2)
            ]
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E3 DIM-configurable three-layer constrained elliptical Gaussian: "
            "fixed HR-pixel centers; bounded area-preserving anisotropy; "
            "density-normalized adaptive-3sigma scatter; unchanged E3 trunk"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
