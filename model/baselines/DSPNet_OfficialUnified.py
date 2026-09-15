"""Unified CAVE training wrapper for the official DSPNet implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_PATH = (
    _PROJECT_ROOT
    / "external_baselines"
    / "DSPNet_official_6a6a065"
    / "CAVE"
    / "DSPNet.py"
)


def _load_upstream_module():
    if not _UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen DSPNet source: {_UPSTREAM_PATH}")
    module_name = "_dspnet_upstream_6a6a065"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DSPNet from {_UPSTREAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    """Expose unchanged official DSPNet through the common trainer API."""

    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        upstream = _load_upstream_module()
        self.net = upstream.DSPNet(num_bands, num_msi)

    def forward(self, lr_hsi, hr_msi, sf=4):
        if int(sf) != 4:
            raise ValueError(
                "The raw official DSPNet forward is fixed to 4x; use the separately "
                "reported frozen adapter for unseen observation factors"
            )
        expected = (int(lr_hsi.shape[-2]) * 4, int(lr_hsi.shape[-1]) * 4)
        if tuple(hr_msi.shape[-2:]) != expected:
            raise ValueError(
                f"Official DSPNet requires HR-MSI grid {expected}, got "
                f"{tuple(hr_msi.shape[-2:])}"
            )
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
