"""Scale-specific training adapter for the frozen official DSPNet source.

The released DSPNet CAVE model hard-codes only its parameter-free LR-HSI
bicubic alignment as ``scale_factor=4``.  This adapter makes that factor an
explicit constructor argument while retaining every learned module and the
rest of the upstream forward graph unchanged.  It is therefore intended for
separately trained DSPNet-4x / DSPNet-8x models, not zero-shot scale transfer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
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


def _make_scale_specific_class(upstream):
    class ScaleSpecificDSPNet(upstream.DSPNet):
        def __init__(self, hschannels, mschannels, training_scale):
            super().__init__(hschannels, mschannels)
            self.training_scale = int(training_scale)

        def forward(self, x, y):
            # This is the sole scale-dependent line in the released forward.
            x = F.interpolate(
                x,
                scale_factor=self.training_scale,
                mode="bicubic",
                align_corners=False,
            )
            if tuple(x.shape[-2:]) != tuple(y.shape[-2:]):
                raise ValueError(
                    f"DSPNet-{self.training_scale}x aligned LR-HSI to "
                    f"{tuple(x.shape[-2:])}, but HR-MSI is {tuple(y.shape[-2:])}"
                )
            x0 = x
            y1 = self.spa1(y)
            y2 = self.spa2(y1)
            y3 = self.spa3(y2)

            z1_2, z1_4, z1_8 = self.spe1(x, x, x)
            z2_2, z2_4, z2_8 = self.spe2(
                self.ds(z1_2), self.ds(z1_4), self.ds(z1_8)
            )
            z3_2, z3_4, z3_8 = self.spe3(
                self.ds(z2_2), self.ds(z2_4), self.ds(z2_8)
            )

            x1 = self.inc(torch.cat((x, y), dim=1))
            x1 = self.mls1(z1_2, z1_4, z1_8, x1)
            x2 = self.down1(torch.cat((x1, y1), dim=1))
            x2 = self.mls2(z2_2, z2_4, z2_8, x2)
            x3 = self.down2(torch.cat((x2, y2), dim=1))
            x3 = self.mls3(z3_2, z3_4, z3_8, x3)
            x = self.up1(torch.cat((x3, y3), dim=1))
            x = self.up2(torch.cat((x, x2, y2), dim=1))
            logits = self.outc(torch.cat((x, x1, y1), dim=1))
            return logits + x0

    return ScaleSpecificDSPNet


class GSFusion(nn.Module):
    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        training_scale: int = 4,
    ):
        super().__init__()
        if int(training_scale) not in {4, 8}:
            raise ValueError("Formal DSPNet scale-specific runs support 4x or 8x")
        upstream = _load_upstream_module()
        net_class = _make_scale_specific_class(upstream)
        self.training_scale = int(training_scale)
        self.net = net_class(num_bands, num_msi, self.training_scale)

    def forward(self, lr_hsi, hr_msi, sf=None):
        if sf is not None and int(sf) != self.training_scale:
            raise ValueError(
                f"This checkpoint is DSPNet-{self.training_scale}x, got sf={sf}"
            )
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
