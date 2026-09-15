"""Unified CAVE/Harvard adapter for the official TGRS 2026 RAMoE."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "RAMoE_official_37b535b"


def _load_official_module():
    name = "_ramoe_official_37b535b"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _OFFICIAL / "RAMoE.py")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load official RAMoE source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    supported_scales = (4, 8, 16, 32)
    inference_adapter = (
        "native official x4 network; frozen unseen scales use explicit "
        "non-learned bicubic alignment to the canonical quarter grid"
    )

    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        module = _load_official_module()
        self.net = module.RAMoEN(
            hsi_channels=num_bands,
            msi_channels=num_msi,
            upscale_factor=4,
            dim_moe=36,
            num_experts=8,
            num_shared=1,
            residual_type="3conv",
            n_feats=96,
            deepth=3,
            value_case="sum",
            act_func="gelu",
            ffn_mode="RAMoE",
        )

    def forward(self, lr_hsi, hr_msi, sf=4):
        scale = int(sf)
        if scale not in self.supported_scales:
            raise RuntimeError(f"Unsupported RAMoE evaluation scale: {scale}")
        canonical_size = (hr_msi.shape[-2] // 4, hr_msi.shape[-1] // 4)
        if tuple(lr_hsi.shape[-2:]) != canonical_size:
            lr_hsi = F.interpolate(
                lr_hsi,
                size=canonical_size,
                mode="bicubic",
                align_corners=False,
            )
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
