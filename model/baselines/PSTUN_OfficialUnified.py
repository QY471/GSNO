"""Minimal single-scale PSTUN-main adapter for unified CAVE training."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "PSTUN_official_925caa6_src" / "PSTUN-main"


def _load_model_class():
    root = str(_OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("architecture.PSTUN")
    return module.PSTUN


class GSFusion(nn.Module):
    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        if num_bands <= 0 or num_msi != 3:
            raise ValueError(f"PSTUN expects positive HSI bands and 3 MSI channels, got {num_bands}/{num_msi}")
        model_class = _load_model_class()
        self.net = model_class(in_channels=num_msi, in_feat=128, out_channels=num_bands, stage=3)

    def forward(self, lr_hsi, hr_msi, sf=4):
        del sf
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
