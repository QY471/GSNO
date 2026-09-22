"""Harvard dataset used by the paper's training protocol.

Adapted from the supplied AFNO Harvard loader; see third_party/README.md.
The loader follows the fixed Harvard protocol:

* 67 training MAT files and 10 testing MAT files are loaded by the caller;
* all 67 loaded training scenes participate in random sampling;
* evaluation uses the first 1024 x 1024 pixels without tiling;
* LR-HSI is generated online from HR-HSI with Gaussian sigma 2.0;
* the sampling phase is ``sf // 2 - 1``;
* the stored fixed-8x ``LRHS`` field is not used;
* a full-resolution coordinate grid is returned as the fourth item.

The dataset path is intentionally not hard-coded here. The training entrypoint
must load the ``HS`` and ``HRMS`` arrays and pass them to ``harvard_dataset``.
"""

import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as tud

from tools.Utils import make_coord, para_setting


LOADED_TRAIN_COUNT = 67
LOADED_TEST_COUNT = 10
SPATIAL_SUPPORT = 1024


def prepare_data_harvard(path, file_num):
    """Load numbered MAT files, reading only ``HS`` and ``HRMS``."""

    hr_hsi = np.zeros((1040, 1392, 31, file_num))
    hr_msi = np.zeros((1040, 1392, 3, file_num))
    for index in range(file_num):
        mat_path = os.path.join(path, f"{index + 1}.mat")
        data = sio.loadmat(mat_path)
        hr_hsi[:, :, :, index] = data["HS"]
        hr_msi[:, :, :, index] = data["HRMS"]
    return hr_hsi, hr_msi


class harvard_dataset(tud.Dataset):
    """Harvard dataset used by the training protocol."""

    source_kind = "harvard_standard"

    def __init__(self, opt, HR_HSI, HR_MSI, istrain=True):
        super().__init__()
        self.path = opt.data_path
        self.istrain = bool(istrain)
        self.factor = int(opt.sf)

        if self.istrain:
            self.num = int(opt.trainset_num)
            self.file_num = LOADED_TRAIN_COUNT
            self.sizeI = int(opt.sizeI)
        else:
            self.num = int(opt.testset_num)
            self.file_num = LOADED_TEST_COUNT
            self.sizeI = int(
                getattr(opt, "eval_crop_size", SPATIAL_SUPPORT)
            )
        self.crop_top = int(getattr(opt, "eval_crop_top", 0)) if not self.istrain else 0
        self.crop_left = int(getattr(opt, "eval_crop_left", 0)) if not self.istrain else 0
        if self.sizeI <= 0:
            raise ValueError(f"Harvard crop size must be positive, got {self.sizeI}")
        if self.crop_top < 0 or self.crop_left < 0:
            raise ValueError("Harvard crop offsets cannot be negative")
        if (
            self.crop_top + self.sizeI > 1040
            or self.crop_left + self.sizeI > 1392
        ):
            raise ValueError(
                "Harvard crop exceeds source bounds: "
                f"top={self.crop_top}, left={self.crop_left}, size={self.sizeI}"
            )

        self.HR_HSI = HR_HSI
        self.HR_MSI = HR_MSI

        if HR_HSI.shape[:3] != (1040, 1392, 31):
            raise ValueError(
                "Expected Harvard HSI shape (1040,1392,31,N), "
                f"got {HR_HSI.shape}"
            )
        if HR_MSI.shape[:3] != (1040, 1392, 3):
            raise ValueError(
                "Expected Harvard MSI shape (1040,1392,3,N), "
                f"got {HR_MSI.shape}"
            )
        if self.istrain and HR_HSI.shape[-1] < self.file_num:
            raise ValueError(
                "The Harvard training split requires all 67 scenes"
            )
        if not self.istrain and HR_HSI.shape[-1] < self.num:
            raise ValueError(
                "The Harvard test split has fewer scenes than testset_num"
            )

    @staticmethod
    def H_z(z, factor, fft_B):
        """Apply Gaussian blur and phase-aligned spatial sampling."""

        frequency = torch.fft.fft2(z, dim=(-2, -1))
        frequency = torch.stack((frequency.real, frequency.imag), -1)

        if len(z.shape) == 3:
            channels, _, _ = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(channels, 1, 1, 1)
            multiplied = torch.cat(
                (
                    (
                        frequency[:, :, :, 0] * fft_B[:, :, :, 0]
                        - frequency[:, :, :, 1] * fft_B[:, :, :, 1]
                    ).unsqueeze(3),
                    (
                        frequency[:, :, :, 0] * fft_B[:, :, :, 1]
                        + frequency[:, :, :, 1] * fft_B[:, :, :, 0]
                    ).unsqueeze(3),
                ),
                3,
            )
            blurred = torch.fft.ifft2(
                torch.complex(multiplied[..., 0], multiplied[..., 1]),
                dim=(-2, -1),
            )
            output = blurred[
                :, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor
            ]
        elif len(z.shape) == 4:
            batch_size, channels, _, _ = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, channels, 1, 1, 1
            )
            multiplied = torch.cat(
                (
                    (
                        frequency[:, :, :, :, 0] * fft_B[:, :, :, :, 0]
                        - frequency[:, :, :, :, 1] * fft_B[:, :, :, :, 1]
                    ).unsqueeze(4),
                    (
                        frequency[:, :, :, :, 0] * fft_B[:, :, :, :, 1]
                        + frequency[:, :, :, :, 1] * fft_B[:, :, :, :, 0]
                    ).unsqueeze(4),
                ),
                4,
            )
            blurred = torch.fft.ifft2(
                torch.complex(multiplied[..., 0], multiplied[..., 1]),
                dim=(-2, -1),
            )
            output = blurred[
                :,
                :,
                int(factor // 2) - 1::factor,
                int(factor // 2) - 1::factor,
            ]
        else:
            raise ValueError(
                f"Expected CHW or BCHW input, got shape {tuple(z.shape)}"
            )

        return output.real

    def __getitem__(self, index):
        if self.istrain:
            scene_index = random.randint(0, self.file_num - 1)
        else:
            scene_index = index

        hr_hsi = self.HR_HSI[:, :, :, scene_index]
        hr_msi = self.HR_MSI[:, :, :, scene_index]

        size = [self.sizeI, self.sizeI]
        fft_B, _ = para_setting("gaussian_blur", self.factor, size, 2.0)
        fft_B = torch.stack(
            (torch.tensor(np.real(fft_B)), torch.tensor(np.imag(fft_B))), dim=2
        ).float()

        if self.istrain:
            px = random.randint(0, SPATIAL_SUPPORT - self.sizeI)
            py = random.randint(0, SPATIAL_SUPPORT - self.sizeI)
        else:
            px = self.crop_top
            py = self.crop_left
        hr_hsi = hr_hsi[px:px + self.sizeI, py:py + self.sizeI, :]
        hr_msi = hr_msi[px:px + self.sizeI, py:py + self.sizeI, :]

        if self.istrain:
            rotations = random.randint(0, 3)
            vertical_flip = random.randint(0, 1)
            horizontal_flip = random.randint(0, 1)

            for _ in range(rotations):
                hr_hsi = np.rot90(hr_hsi)
                hr_msi = np.rot90(hr_msi)
            for _ in range(vertical_flip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()
            for _ in range(horizontal_flip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = (
            torch.tensor(hr_hsi.copy(), dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        hr_msi = (
            torch.tensor(hr_msi.copy(), dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B).float()

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)
        coord = make_coord((self.sizeI, self.sizeI), flatten=False)
        return lr_hsi, hr_msi, hr_hsi, coord

    def __len__(self):
        return self.num
