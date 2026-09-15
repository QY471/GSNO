"""E3 ablation: use three sequential current Gaussian residual layers."""

from model.GSFusion_E3_GaussianMechanismAblationCommon import (
    ThreeLayerGSFusion as GSFusion,
    compute_loss,
    sam_loss,
)

__all__ = ["GSFusion", "compute_loss", "sam_loss"]
