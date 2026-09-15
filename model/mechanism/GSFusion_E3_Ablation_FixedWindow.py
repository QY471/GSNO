"""E3 ablation: replace adaptive 3-sigma support with raster_ratio=0.1."""

from model.GSFusion_E3_GaussianMechanismAblationCommon import (
    FixedWindowGSFusion as GSFusion,
    compute_loss,
    sam_loss,
)

__all__ = ["GSFusion", "compute_loss", "sam_loss"]
