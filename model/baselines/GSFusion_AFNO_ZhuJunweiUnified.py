"""ZhuJunwei AFNO adapted to the unified GSFusion training interface."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.Utils import make_coord


class ADCILayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True, unbiased=False)
        return self.weight * ((inputs - mean) / (std + self.eps)) + self.bias


class ADCI(nn.Module):
    def __init__(self, in_channels, mlp_hidden_dim):
        super().__init__()
        self.qkv_conv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            ADCILayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.padding = nn.ReflectionPad2d(1)

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        query, key, value = torch.chunk(self.qkv_conv(inputs), chunks=3, dim=1)
        key_neighbors = F.unfold(self.padding(key), kernel_size=3).view(
            batch, channels, 9, height, width
        )
        value_neighbors = F.unfold(self.padding(value), kernel_size=3).view(
            batch, channels, 9, height, width
        )
        query_minus_key = query.unsqueeze(2) - key_neighbors
        query_minus_key = query_minus_key.permute(0, 3, 4, 2, 1).contiguous()
        attention_scores = F.softmax(self.mlp(query_minus_key), dim=-2)
        value_neighbors = value_neighbors.permute(0, 3, 4, 2, 1).contiguous()
        weighted_value = torch.sum(value_neighbors * attention_scores, dim=3)
        weighted_value = weighted_value.permute(0, 3, 1, 2).contiguous()
        return weighted_value + self.gate(inputs)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.weight * ((inputs - mean) / (std + self.eps)) + self.bias


class SimpleAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.head_channels = channels // heads
        self.heads = heads
        self.channels = channels
        self.qkv_proj = nn.Conv2d(channels, 3 * channels, 1)
        self.o_proj1 = nn.Conv2d(channels, channels, 1)
        self.o_proj2 = nn.Conv2d(channels, channels, 1)
        self.kln = LayerNorm((heads, 1, self.head_channels))
        self.vln = LayerNorm((heads, 1, self.head_channels))
        self.act = nn.GELU()

    def forward(self, inputs, name="0"):
        del name
        batch, channels, height, width = inputs.shape
        residual = inputs
        qkv = self.qkv_proj(inputs).permute(0, 2, 3, 1).reshape(
            batch, height * width, self.heads, 3 * self.head_channels
        )
        query, key, value = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        key = self.kln(key)
        value = self.vln(value)
        value = torch.matmul(key.transpose(-2, -1), value) / (height * width)
        value = torch.matmul(query, value)
        value = value.permute(0, 2, 1, 3).reshape(
            batch, height, width, channels
        )
        output = self.o_proj1(value.permute(0, 3, 1, 2) + residual)
        output = F.interpolate(
            output, scale_factor=2, mode="bicubic", align_corners=False
        )
        output = self.act(output)
        output = F.interpolate(
            output, scale_factor=0.5, mode="bicubic", align_corners=False
        )
        output = self.o_proj2(output)
        return output + residual


class GSFusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        del kwargs
        self.shallow_encoder1 = nn.Sequential(nn.Conv2d(31, 32, 1))
        self.shallow_encoder2 = nn.Sequential(nn.Conv2d(34, 32, 1))
        self.conv0 = SimpleAttention(32, 8)
        self.conv1 = SimpleAttention(32, 8)
        self.convw = nn.Conv2d(66, 32, 1)
        self.conv00 = nn.Conv2d(160, 64, 1)
        self.conv01 = nn.Conv2d(64, 32, 1)
        self.act = nn.ReLU()
        self.project = nn.Sequential(
            nn.Conv2d(32, 31, 1),
            nn.GELU(),
            nn.Conv2d(31, 31, 1),
        )
        self.ADCI1_1 = ADCI(32, 32)
        self.ADCI1_2 = ADCI(32, 32)
        self.ADCI1_3 = ADCI(32, 32)
        self.ADCI2_1 = ADCI(32, 32)
        self.ADCI2_2 = ADCI(32, 32)
        self.ADCI2_3 = ADCI(32, 32)

    def infi(self, features, high_resolution_features, coordinates):
        batch = features.shape[0]
        height, width = features.shape[-2:]
        feature_coordinates = make_coord(
            (height, width), flatten=False
        ).to(features.device).permute(2, 0, 1).unsqueeze(0).expand(
            batch, 2, height, width
        )
        radius_x = 1 / height
        radius_y = 1 / width
        candidates = []
        areas = []
        for offset_x in (-1, 1):
            for offset_y in (-1, 1):
                shifted = coordinates.clone()
                shifted[:, :, 0] += offset_x * radius_x
                shifted[:, :, 1] += offset_y * radius_y
                query_features = F.grid_sample(
                    features, shifted.flip(-1), mode="nearest", align_corners=False
                )
                query_coordinates = F.grid_sample(
                    feature_coordinates,
                    shifted.flip(-1),
                    mode="nearest",
                    align_corners=False,
                )
                relative = coordinates.permute(0, 3, 1, 2) - query_coordinates
                relative[:, 0] *= height
                relative[:, 1] *= width
                areas.append(torch.abs(relative[:, 0] * relative[:, 1]) + 1e-9)
                candidates.append(
                    self.convw(
                        torch.cat(
                            (query_features, relative),
                            dim=1,
                        )
                    )
                )

        total_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]
        weighted = [
            candidate * (area / total_area).unsqueeze(1)
            for candidate, area in zip(candidates, areas)
        ]
        output = self.conv00(
            torch.cat((torch.cat(weighted, dim=1), high_resolution_features), dim=1)
        )
        output = F.interpolate(
            output, scale_factor=2, mode="bicubic", align_corners=False
        )
        output = self.act(output)
        output = F.interpolate(
            output, scale_factor=0.5, mode="bicubic", align_corners=False
        )
        output = self.conv01(output)
        output = self.conv1(self.conv0(output, 0), 1)
        return self.project(output)

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf: Optional[int] = None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del return_aux
        if sf is None:
            if hr_msi.shape[-2] % lr_hsi.shape[-2] != 0:
                raise ValueError("AFNO requires an integer HSI/MSI scale ratio")
            sf = hr_msi.shape[-2] // lr_hsi.shape[-2]
        if output_size is not None and tuple(output_size) != tuple(hr_msi.shape[-2:]):
            raise ValueError("unified AFNO only predicts on the native HR-MSI grid")
        lr_hsi_up = F.interpolate(
            lr_hsi, scale_factor=sf, mode="bicubic", align_corners=False
        )
        if lr_hsi_up.shape[-2:] != hr_msi.shape[-2:]:
            raise ValueError(
                "AFNO requires sf to map LR-HSI exactly onto the HR-MSI grid"
            )
        high_resolution_features = self.shallow_encoder2(
            torch.cat((hr_msi, lr_hsi_up), dim=1)
        )
        low_resolution_features = self.shallow_encoder1(lr_hsi)
        high_resolution_features = self.ADCI1_3(
            self.ADCI1_2(self.ADCI1_1(high_resolution_features))
        )
        low_resolution_features = self.ADCI2_3(
            self.ADCI2_2(self.ADCI2_1(low_resolution_features))
        )
        high_resolution_features_down = F.interpolate(
            high_resolution_features,
            size=low_resolution_features.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        query_features = torch.cat(
            (high_resolution_features_down, low_resolution_features), dim=1
        )
        batch, _, output_height, output_width = hr_msi.shape
        coordinates = make_coord(
            (output_height, output_width), flatten=False
        ).to(hr_msi.device).unsqueeze(0).expand(
            batch, output_height, output_width, 2
        )
        return (
            self.infi(
                query_features, high_resolution_features, coordinates
            )
            + lr_hsi_up
        )


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
