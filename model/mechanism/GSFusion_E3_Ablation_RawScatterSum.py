"""E3 ablation: remove density normalization from Gaussian scatter."""

from model.GSFusion_E3_GaussianMechanismAblationCommon import (
    RawSumGSFusion as GSFusion,
    compute_loss,
    sam_loss,
)

__all__ = ["GSFusion", "compute_loss", "sam_loss"]
