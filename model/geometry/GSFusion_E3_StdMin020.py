"""E3 diagnostic with a lower minimum circular Gaussian standard deviation.

This keeps the complete E3 architecture and parameter initialization order
unchanged.  The only experimental change is ``std_min_px: 0.30 -> 0.20``;
``std_max_px`` remains 1.50 HR pixels.
"""

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as _E3GSFusion,
)
from model.GSFusion_GSNO import compute_loss, sam_loss


class GSFusion(_E3GSFusion):
    """Original E3 with a 0.20-pixel lower bound for circular std."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_refine.std_min_px = 0.20
        self.gaussian_refine.std_max_px = 1.50
        self.arch_summary += "; diagnostic std range=[0.20,1.50] HR pixels"

