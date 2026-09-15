"""Fixed-cap wrapper for frozen evaluation of the Chikusei gated-anchor model."""

from model.backbones.GSFusion_E3_ChikuseiGatedSpectralAnchorGaussianADCICUDAExactContinuous import (
    GSFusion as UncappedGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(UncappedGSFusion):
    def __init__(self, *args, gaussian_rms_cap: float = 0.2, **kwargs):
        super().__init__(*args, gaussian_rms_cap=gaussian_rms_cap, **kwargs)


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
