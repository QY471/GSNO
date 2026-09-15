"""E3 ablation: make std scale with the current HR canvas."""

from model.GSFusion_E3_GaussianMechanismAblationCommon import (
    CanvasScaledStdGSFusion as GSFusion,
    compute_loss,
    sam_loss,
)

__all__ = ["GSFusion", "compute_loss", "sam_loss"]
