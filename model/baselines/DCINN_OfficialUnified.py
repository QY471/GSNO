"""Unified-protocol adapter for the official IJCV DCINN HMF model.

DCINN's released HMF entrypoint assumes a Nikon-D700 spectral response and
uses its fixed pseudo-inverse to lift the observed three-channel image to 31
bands.  In the unified experiments the HR-MSI is the project's supplied
paired observation, so using the D700 inverse would mix sensor protocols.
This adapter deterministically estimates the paired response from *training*
HSI/MSI pairs only, freezes its pseudo-inverse as a buffer, and keeps the
official DCINN backbone unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "DCINN_official_d859e92"


def _load_official_module():
    root = str(_OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "_dcinn_official_d859e92_hmf"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _OFFICIAL / "model" / "dcinn_hmf.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load official DCINN HMF model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fit_paired_response(dataset: str, data_path: Path):
    """Fit the dataset response using one deterministic training scene.

    The stored paired MSI is an exact fixed linear projection of the HSI for
    both supported datasets.  A dense, regularly subsampled training scene is
    therefore sufficient to recover the 31x3 response to numerical precision.
    No validation or test data is read here.
    """

    data_path = data_path.resolve()
    if dataset == "cave":
        split_path = data_path / "Train.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing CAVE training split: {split_path}")
        names = [name for name in split_path.read_text().splitlines() if name]
        if not names:
            raise ValueError(f"Empty CAVE training split: {split_path}")
        hsi = sio.loadmat(data_path / "HSI" / f"{names[0]}.mat")["hsi"]
        msi = sio.loadmat(data_path / "RGB" / f"{names[0]}.mat")["rgb"]
        stride = 2
    else:
        mat_path = data_path / "1.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing Harvard training scene: {mat_path}")
        paired = sio.loadmat(mat_path)
        hsi, msi = paired["HS"], paired["HRMS"]
        stride = 4

    hsi_samples = np.asarray(
        hsi[::stride, ::stride, :], dtype=np.float64
    ).reshape(-1, 31)
    msi_samples = np.asarray(
        msi[::stride, ::stride, :], dtype=np.float64
    ).reshape(-1, 3)
    normal = hsi_samples.T @ hsi_samples
    cross = hsi_samples.T @ msi_samples
    ridge = 1e-12 * float(np.trace(normal)) / normal.shape[0]
    response = np.linalg.solve(normal + ridge * np.eye(31), cross)
    fitted = hsi_samples @ response
    relative_error = float(
        np.linalg.norm(fitted - msi_samples) / np.linalg.norm(msi_samples)
    )
    if not np.isfinite(relative_error) or relative_error > 1e-6:
        raise ValueError(
            "Paired HSI/MSI is not explained by one fixed response: "
            f"dataset={dataset} relative_error={relative_error:.3e}"
        )
    response_inverse = np.linalg.pinv(response)
    return (
        np.asarray(response, dtype=np.float32),
        np.asarray(response_inverse, dtype=np.float32),
        relative_error,
    )


class GSFusion(nn.Module):
    """Expose official DCINN through the common ``(LR-HSI, HR-MSI, sf)`` API."""

    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        dataset: str = "cave",
        calibration_data_path: str | Path | None = None,
    ):
        super().__init__()
        if (num_bands, num_msi) != (31, 3):
            raise ValueError("Official DCINN HMF requires 31 HSI and 3 MSI channels")
        dataset = dataset.lower()
        if dataset not in {"cave", "harvard"}:
            raise ValueError(f"Unsupported DCINN calibration dataset: {dataset!r}")
        if calibration_data_path is None:
            raise ValueError(
                "DCINN unified calibration_data_path is required; the D700 "
                "inverse must not be used with project-paired HR-MSI"
            )
        module = _load_official_module()
        self.net = module.DCINN(channel_in=31, channel_out=31, block_num=4)
        response, response_inverse, relative_error = _fit_paired_response(
            dataset, Path(calibration_data_path)
        )
        self.dataset = dataset
        self.calibration_data_path = str(Path(calibration_data_path).resolve())
        self.register_buffer("response", torch.from_numpy(response))
        self.register_buffer(
            "response_inverse", torch.from_numpy(response_inverse).unsqueeze(0)
        )
        self.register_buffer(
            "response_calibration_relative_error",
            torch.tensor(relative_error, dtype=torch.float32),
        )

    def forward(self, lr_hsi, hr_msi, sf=4):
        del sf
        lms = F.interpolate(
            lr_hsi,
            size=hr_msi.shape[-2:],
            # Match the released DCINN HMF training entrypoint exactly.
            mode="bilinear",
            align_corners=False,
        )
        batch, _, height, width = hr_msi.shape
        msi_spectral = rearrange(hr_msi, "b c h w -> b (h w) c")
        msi_spectral = torch.matmul(msi_spectral, self.response_inverse)
        msi_spectral = rearrange(
            msi_spectral, "b (h w) c -> b c h w", h=height, w=width
        )
        return self.net(msi_spectral - lms, lms, hr_msi) + lms


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
