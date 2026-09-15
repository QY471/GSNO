"""Formal B0/B1 ablations of the strong GSNO backbone.

Both variants keep the same HSI/MSI encoders, bicubic latent lifting, fusion,
decoder, and raw-LR bicubic residual base used by the strong GSNO model.  The
only controlled variable is the three-block HR refinement stage:

* ``identity``: no HR refinement (B0-Formal).
* ``ffn_only``: the residual 1x1 FFN from each original GSNO Gaussian block,
  without Gaussian neighborhood gathering (B1-Formal).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


class ResidualFFNBlock(nn.Module):
    """The FFN sub-block of the original ``GaussianSplatEncoder``."""

    def __init__(self, dim: int):
        super().__init__()
        # Keep the name and layer layout identical to the B2 GSNO block so its
        # initial FFN weights can be copied exactly for a fair ablation.
        self.ffd = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim * 4, dim, 1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.ffd(x)
        out = x + update
        if not torch.jit.is_scripting():
            with torch.no_grad():
                self.last_stats = {
                    "ffn_input_abs_mean": float(x.detach().abs().mean()),
                    "ffn_update_abs_mean": float(update.detach().abs().mean()),
                    "ffn_update_input_ratio": float(
                        update.detach().abs().mean()
                        / (x.detach().abs().mean() + 1e-8)
                    ),
                }
        return out


class GSFusion(nn.Module):
    """Strong GSNO backbone with Identity or residual-FFN HR refinement."""

    VALID_REFINE_MODES = {"identity", "ffn_only"}

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        num_gs_layers: int = 3,
        adci_layers: int = 3,
        refine_mode: str = "identity",
        **_: object,
    ):
        super().__init__()
        if refine_mode not in self.VALID_REFINE_MODES:
            raise ValueError(
                f"Unknown refine_mode={refine_mode}; "
                f"expected one of {sorted(self.VALID_REFINE_MODES)}"
            )

        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_gs_layers = int(num_gs_layers)
        self.refine_mode = refine_mode

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        if refine_mode == "ffn_only":
            self.gs_layers: nn.Module = nn.ModuleList(
                [ResidualFFNBlock(dim) for _ in range(num_gs_layers)]
            )
        else:
            self.gs_layers = nn.Identity()

        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        refine_name = (
            f"ResidualFFN x{num_gs_layers}"
            if refine_mode == "ffn_only"
            else "Identity"
        )
        self.arch_summary = (
            "GSNO encoders + bicubic latent lifting + 1x1 fusion + "
            f"{refine_name} + residual decoder; dim={dim}, "
            f"ADCI={adci_layers} per stream"
        )

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        if self.refine_mode == "identity":
            return []
        stats = []
        for index, layer in enumerate(self.gs_layers):
            layer_stats = getattr(layer, "last_stats", None)
            if layer_stats is not None:
                stats.append({"layer": f"ffn_{index}", **layer_stats})
        return stats

    def forward(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None
    ) -> torch.Tensor:
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        feat = self.conv0(torch.cat([f_hsi_hr, f_msi], dim=1))
        if self.refine_mode == "ffn_only":
            for layer in self.gs_layers:
                feat = layer(feat)

        residual = self.fc2(F.gelu(self.fc1(feat)))
        return base + residual


__all__ = ["GSFusion", "ResidualFFNBlock", "compute_loss", "sam_loss"]
