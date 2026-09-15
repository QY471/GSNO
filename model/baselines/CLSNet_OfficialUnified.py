"""Unified CAVE/Harvard adapter for the official Information Fusion 2026 CLSNet.

Upstream: https://github.com/HengYang01/CLSNet
Commit: 6b13371297a78922c1d7fd3096fdf1e51f5f7192
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "CLSNet_official_6b13371"


def _load_official_module():
    name = "_clsnet_official_6b13371"
    if name in sys.modules:
        return sys.modules[name]
    source = _OFFICIAL / "model" / "CLSNet.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen CLSNet source: {source}")

    text = source.read_text(encoding="utf-8")
    marker = "from thop import profile"
    if marker not in text:
        raise RuntimeError("CLSNet release layout changed; unguarded demo marker is missing")
    # The released file runs a CUDA/THOP demo at import time.  Exclude only that
    # unguarded demo; all class definitions above it remain byte-for-byte upstream.
    class_source = text.split(marker, 1)[0]
    module = types.ModuleType(name)
    module.__file__ = str(source)
    sys.modules[name] = module
    exec(compile(class_source, str(source), "exec"), module.__dict__)
    return module


class GSFusion(nn.Module):
    """Expose the official CLSNet core through the common fusion interface."""

    supported_scales = (4, 8, 16, 32)
    inference_adapter = (
        "official CLSNet learnable core; released parameter-free nearest LR-HSI "
        "alignment is evaluated at the actual HR-MSI target grid"
    )

    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        training_scale: int = 4,
    ):
        super().__init__()
        if (num_bands, num_msi) != (31, 3):
            raise ValueError("Official CLSNet CAVE/Harvard model requires 31/3 channels")
        if int(training_scale) not in self.supported_scales:
            raise ValueError(f"Unsupported CLSNet training scale: {training_scale}")
        self.training_scale = int(training_scale)
        module = _load_official_module()
        self.net = module.CLSNet(in_channels=num_msi, out_channels=num_bands)
        # Replace only the released fixed nearest x8 operation.  The wrapper
        # performs the identical parameter-free operation at the actual target size.
        self.net.upsamp = nn.Identity()

    def reset_custom_init(self):
        """Reproduce the explicit initialization in the released trainers."""
        for module in self.net.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

    def forward(self, lr_hsi, hr_msi, sf=4):
        scale = int(sf)
        if scale not in self.supported_scales:
            raise RuntimeError(f"Unsupported CLSNet scale: {scale}")
        expected = (lr_hsi.shape[-2] * scale, lr_hsi.shape[-1] * scale)
        if tuple(hr_msi.shape[-2:]) != expected:
            raise RuntimeError(
                f"CLSNet x{scale} expects HR-MSI grid {expected}, got "
                f"{tuple(hr_msi.shape[-2:])}"
            )
        if hr_msi.shape[-2] % 8 or hr_msi.shape[-1] % 8:
            raise RuntimeError("CLSNet HR-MSI tile dimensions must be divisible by 8")
        aligned_hsi = F.interpolate(lr_hsi, size=hr_msi.shape[-2:], mode="nearest")
        return self.net(aligned_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
