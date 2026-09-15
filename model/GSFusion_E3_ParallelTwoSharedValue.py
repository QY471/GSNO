"""Two parallel constrained Gaussian experts with one shared value head."""

from __future__ import annotations

from model.GSFusion_E3_ParallelTwoConstrainedEllipticalGaussian import (
    GSFusion as ParallelTwoGaussianGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ParallelTwoGaussianGSFusion):
    """Keep independent geometry while tying both experts' value parameters."""

    variant_name = "two_parallel_constrained_elliptical_gaussian_shared_value"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussian_refine_extra[0].residual_value_head = (
            self.gaussian_refine.residual_value_head
        )
        self.arch_summary = (
            "E3 DIM-configurable two-expert parallel constrained elliptical "
            "Gaussian mixture: independent opacity and geometry, one tied "
            "value head, normalized global mixing, no state cascade"
        )


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
