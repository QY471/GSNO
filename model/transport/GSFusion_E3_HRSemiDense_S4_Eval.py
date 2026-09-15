"""Evaluation adapter for the E3 HR-semidense stride-4 checkpoint."""

from model.transport.GSFusion_HRSemiDenseGaussian import GSFusion as _SemiDenseGSFusion


class GSFusion(_SemiDenseGSFusion):
    """Construct the semidense model with its formal stride-4 setting."""

    def __init__(self, *args, **kwargs):
        kwargs["anchor_stride_hr"] = 4
        super().__init__(*args, **kwargs)
