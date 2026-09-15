"""Unified HSI-MSI interface adapter for the official IJCV 2025 LRTN.

The official model is spatial-size dependent and evaluates 64x64 tiles. Its
LR-HSI pyramid alignment is parameter-free, so a frozen x4 checkpoint can be
evaluated at altered LR-HSI observation resolutions by aligning the LR-HSI to
the same fixed 64/32/16 feature grids. No learned module or output resize is
introduced.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "LRTN_official_00231bf_src"


def _load_official_module():
    root = str(_OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    if "thop" not in sys.modules:
        thop = types.ModuleType("thop")
        thop.profile = None
        thop.clever_format = None
        sys.modules["thop"] = thop
    name = "_lrtn_official_00231bf"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _OFFICIAL / "LRTN.py")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load official LRTN")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    supported_scales = (4, 8, 16, 32)
    inference_adapter = (
        "direct altered-observation inference with the official parameter-free "
        "LR-HSI pyramid aligned to fixed 64/32/16 feature grids"
    )

    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        if num_msi != 3 or num_bands <= 0:
            raise ValueError(
                f"LRTN requires a positive HSI band count and 3 MSI channels, "
                f"got {num_bands}/{num_msi}"
            )
        module = _load_official_module()
        self.net = module.Cross_Guide_Fusion(num_bands, 64, 64, 1)
        # Official LRTN always consumes LR-HSI features at the 64/32/16 grids.
        # Using explicit target sizes preserves the x4 training path and also
        # permits frozen x8/x16/x32 observations without changing any weights.
        self.net.hsi_up1 = nn.Upsample(size=(64, 64), mode="bilinear")
        self.net.hsi_up2 = nn.Upsample(size=(32, 32), mode="bilinear")
        self.net.hsi_up3 = nn.Upsample(size=(16, 16), mode="bilinear")

    def reset_custom_init(self):
        """Reproduce the explicit initialization in the official trainer."""
        for module in self.net.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

    def forward(self, lr_hsi, hr_msi, sf=4):
        if int(sf) not in self.supported_scales:
            raise RuntimeError(f"Unsupported LRTN evaluation scale: {sf}")
        height, width = map(int, hr_msi.shape[-2:])
        if height > 64 or width > 64:
            raise RuntimeError(
                "Official LRTN has learned spatial-size-dependent parameters; use aligned 64x64 tiling"
            )
        if (height, width) == (64, 64):
            return self.net(lr_hsi, hr_msi)

        # Chikusei is 680x680, so its final aligned tile is smaller than 64.
        # Pad only that boundary tile to the official operating grid, evaluate
        # the unchanged network, and crop the prediction back.
        padded_hr = F.pad(
            hr_msi, (0, 64 - width, 0, 64 - height), mode="replicate"
        )
        target_lr = 64 // int(sf)
        lr_height, lr_width = map(int, lr_hsi.shape[-2:])
        padded_lr = F.pad(
            lr_hsi,
            (0, target_lr - lr_width, 0, target_lr - lr_height),
            mode="replicate",
        )
        return self.net(padded_lr, padded_hr)[..., :height, :width]


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
