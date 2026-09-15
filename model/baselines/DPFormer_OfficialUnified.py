"""Unified CAVE/Harvard adapter for the official Pattern Recognition 2026 DPFormer.

Upstream: https://github.com/cvmdsp/DPFormer
Commit: a47f90d6ef99be8e634e463895b70d23aaaa5586
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "DPFormer_official_a47f90d"


def _install_basicsr_compatibility() -> None:
    """Supply only the two BasicSR utilities used by the released model file."""

    if "basicsr.utils.registry" not in sys.modules:
        registry_module = types.ModuleType("basicsr.utils.registry")

        class _Registry:
            def register(self):
                return lambda cls: cls

        registry_module.ARCH_REGISTRY = _Registry()
        sys.modules.setdefault("basicsr", types.ModuleType("basicsr"))
        sys.modules.setdefault("basicsr.utils", types.ModuleType("basicsr.utils"))
        sys.modules["basicsr.utils.registry"] = registry_module

    if "basicsr.archs.arch_util" not in sys.modules:
        arch_util = types.ModuleType("basicsr.archs.arch_util")
        arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
        arch_util.trunc_normal_ = nn.init.trunc_normal_
        sys.modules.setdefault("basicsr.archs", types.ModuleType("basicsr.archs"))
        sys.modules["basicsr.archs.arch_util"] = arch_util


def _load_official_module():
    name = "_dpformer_official_a47f90d"
    if name in sys.modules:
        return sys.modules[name]
    _install_basicsr_compatibility()
    source = _OFFICIAL / "DPFormer.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen DPFormer source: {source}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load DPFormer from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    """Preserve the official learnable architecture with target-grid alignment."""

    supported_scales = (4, 8, 16, 32)
    inference_adapter = (
        "official DPFormer learnable core; released parameter-free bilinear LR-HSI "
        "alignment is evaluated at the actual HR-MSI target grid"
    )

    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        training_scale: int = 4,
    ):
        super().__init__()
        if num_bands <= 0 or num_msi != 3:
            raise ValueError(f"DPFormer expects positive HSI bands and 3 MSI channels, got {num_bands}/{num_msi}")
        if int(training_scale) not in self.supported_scales:
            raise ValueError(f"Unsupported DPFormer training scale: {training_scale}")
        self.training_scale = int(training_scale)
        module = _load_official_module()
        self.net = module.DPFormer(
            self.training_scale,
            num_msi,
            num_bands,
            img_size=64,
        )

    def forward(self, lr_hsi, hr_msi, sf=4):
        scale = int(sf)
        if scale not in self.supported_scales:
            raise RuntimeError(f"Unsupported DPFormer scale: {scale}")
        expected = (lr_hsi.shape[-2] * scale, lr_hsi.shape[-1] * scale)
        if tuple(hr_msi.shape[-2:]) != expected:
            raise RuntimeError(
                f"DPFormer x{scale} expects HR-MSI grid {expected}, got "
                f"{tuple(hr_msi.shape[-2:])}"
            )
        # This released attribute controls only F.interpolate; it owns no weights.
        self.net.scale_ratio = scale
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    spatial = F.l1_loss(prediction, target)
    frequency = torch.mean(torch.abs(torch.fft.fft2(prediction) - torch.fft.fft2(target)))
    return spatial + 0.1 * frequency


__all__ = ["GSFusion", "compute_loss"]
