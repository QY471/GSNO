"""Chikusei loader for 4x-only training and 4/8/16/32x evaluation.

Only GT and RGB are read from HDF5. LR-HSI is generated online with the
same Gaussian degradation convention used by the supplied AFNO-style code.
"""

from __future__ import annotations

import random

import h5py
import numpy as np
import torch
import torch.utils.data as tud

from tools.Utils import make_coord, para_setting


class ChikuseiAFNODataset(tud.Dataset):
    def __init__(self, opt, _hr_hsi=None, _hr_msi=None, istrain=True):
        super().__init__()
        if isinstance(opt, (str, bytes)):
            # Lightweight direct-use compatibility: Dataset(path, sf, istrain).
            self.file_path = str(opt)
            self.factor = int(_hr_hsi)
            if isinstance(_hr_msi, bool):
                istrain = _hr_msi
        else:
            self.file_path = str(opt.data_path if istrain else opt.test_data_path)
            self.factor = int(opt.sf)
            self.augment = bool(getattr(opt, "chikusei_augment", True))
        self.istrain = bool(istrain)
        if not hasattr(self, "augment"):
            self.augment = True
        self._file = None
        self._gt = None
        self._rgb = None
        with h5py.File(self.file_path, "r") as handle:
            self.length = int(handle["GT"].shape[0])
            self.gt_shape = tuple(handle["GT"].shape)
            self.rgb_shape = tuple(handle["RGB"].shape)
        if self.gt_shape[1] != 128 or self.rgb_shape[1] != 3:
            raise ValueError(
                f"Expected GT/RGB channels 128/3, got {self.gt_shape}/{self.rgb_shape}"
            )

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
            self._gt = self._file["GT"]
            self._rgb = self._file["RGB"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_gt"] = None
        state["_rgb"] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._gt = None
            self._rgb = None

    def __del__(self):
        self.close()

    @staticmethod
    def _degrade(gt, factor):
        height, width = gt.shape[-2:]
        fft_b, _ = para_setting("gaussian_blur", factor, [height, width], 2.0)
        kernel = torch.stack(
            (torch.tensor(np.real(fft_b)), torch.tensor(np.imag(fft_b))), dim=2
        ).float()
        frequency = torch.fft.fft2(gt, dim=(-2, -1))
        kernel_complex = torch.complex(kernel[..., 0], kernel[..., 1])
        blurred = torch.fft.ifft2(
            frequency * kernel_complex.unsqueeze(0), dim=(-2, -1)
        ).real
        phase = factor // 2 - 1
        return blurred[:, phase::factor, phase::factor]

    def __getitem__(self, index):
        self._ensure_open()
        gt = np.asarray(self._gt[index], dtype=np.float32)
        rgb = np.asarray(self._rgb[index], dtype=np.float32)

        if self.istrain and self.augment:
            rotations = random.randint(0, 3)
            if rotations:
                gt = np.rot90(gt, rotations, axes=(-2, -1)).copy()
                rgb = np.rot90(rgb, rotations, axes=(-2, -1)).copy()
            if random.randint(0, 1):
                gt = gt[:, :, ::-1].copy()
                rgb = rgb[:, :, ::-1].copy()
            if random.randint(0, 1):
                gt = gt[:, ::-1, :].copy()
                rgb = rgb[:, ::-1, :].copy()

        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        lr_hsi = self._degrade(gt, self.factor)
        coord = make_coord(gt.shape[-2:], flatten=False)
        return lr_hsi, rgb, gt, coord

    def __len__(self):
        return self.length


# Compatibility name for training code that expects a function-like class.
chikusei_dataset = ChikuseiAFNODataset
