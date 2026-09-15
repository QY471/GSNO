"""Unified-protocol wrapper for the official AAAI 2025 OTIAS model.

The learnable OTIAS architecture is loaded unchanged from the frozen upstream
snapshot at commit d4c2014a8af76cb7ee9fb2ce3a92267776f885ec.  The project
supplies its own paired HR-MSI and FFT/Gaussian LR-HSI observations.

The released constructor stores patch-size-dependent AdaptiveMaxPool2d output
sizes.  Those pooling layers have no parameters, so before every forward pass
we set their output sizes from the actual HR-MSI grid.  This reproduces the
released 64x64-patch path exactly and permits the same learned weights to run
on full images and other reference grids without changing learned parameters.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = (
    _PROJECT_ROOT / "external_baselines" / "OTIAS_official_d4c2014"
)
_UPSTREAM_MODEL = _UPSTREAM_ROOT / "model" / "otias.py"


def _install_optional_import_shims() -> None:
    """Supply modules used only by the upstream profiling example."""
    if "fvcore.nn" not in sys.modules:
        fvcore = types.ModuleType("fvcore")
        fvcore_nn = types.ModuleType("fvcore.nn")
        fvcore_nn.FlopCountAnalysis = None
        fvcore_nn.flop_count_table = None
        fvcore.nn = fvcore_nn
        sys.modules["fvcore"] = fvcore
        sys.modules["fvcore.nn"] = fvcore_nn

    if "timm.models.layers" not in sys.modules:
        timm = types.ModuleType("timm")
        timm_models = types.ModuleType("timm.models")
        timm_layers = types.ModuleType("timm.models.layers")
        timm_layers.DropPath = nn.Identity
        timm_layers.to_2tuple = lambda value: (value, value)
        timm_layers.trunc_normal_ = nn.init.trunc_normal_
        timm.models = timm_models
        timm_models.layers = timm_layers
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = timm_models
        sys.modules["timm.models.layers"] = timm_layers

    if "thop" not in sys.modules:
        thop = types.ModuleType("thop")
        thop.profile = None
        sys.modules["thop"] = thop


def _load_upstream_module():
    if not _UPSTREAM_MODEL.is_file():
        raise FileNotFoundError(f"Missing frozen OTIAS source: {_UPSTREAM_MODEL}")

    module_name = "_otias_upstream_d4c2014"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _install_optional_import_shims()
    model_dir = str(_UPSTREAM_MODEL.parent)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location(module_name, _UPSTREAM_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OTIAS from {_UPSTREAM_MODEL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    """Expose official OTIAS through the common ``(LR, HR-MSI, sf)`` API."""

    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        dataset: str = "cave",
    ):
        super().__init__()
        if num_bands != 31 or num_msi != 3:
            raise ValueError("Official OTIAS requires 31 HSI and 3 MSI channels")
        dataset = dataset.lower()
        if dataset not in {"cave", "harvard"}:
            raise ValueError(f"Official OTIAS wrapper does not support dataset={dataset!r}")
        upstream = _load_upstream_module()
        network_class = (
            upstream.otias_c_x4 if dataset == "cave" else upstream.otias_h_x4
        )
        self.dataset = dataset
        self.net = network_class(
            n_select_bands=num_msi,
            n_bands=num_bands,
            feat_dim=128,
            guide_dim=128,
            sz=64,
        )

    def forward(self, lr_hsi, hr_msi, sf=4):
        del sf
        height, width = (int(value) for value in hr_msi.shape[-2:])
        if height < 4 or width < 4:
            raise ValueError(f"OTIAS requires an HR grid of at least 4x4, got {height}x{width}")

        # These modules contain no parameters or buffers.  At 64x64 this is
        # exactly the released (32x32, 16x16) pooling configuration.
        self.net.down32 = nn.AdaptiveMaxPool2d((height // 2, width // 2))
        self.net.down16 = nn.AdaptiveMaxPool2d((height // 4, width // 4))
        lms = F.interpolate(
            lr_hsi,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return self.net(hr_msi, lms, lr_hsi)


def compute_loss(prediction, target, epoch, **kwargs):
    """The released OTIAS training objective is mean absolute error."""
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
