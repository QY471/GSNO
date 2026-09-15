"""E6 with LR-cell Gaussian primitives directly transported to the HR grid."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss
from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    LRCellGaussianTransportDualSource,
    MSIGuidedHSILocalRouting,
    ScaleSharedMSIFootprintSampler,
)


class GSFusion(nn.Module):
    """Change only E6's Gaussian grid from HR pixels to LR-HSI cells."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        footprint_mode: str = "point",
        gaussian_enabled: bool = True,
        gaussian_alpha: float = 1.0,
        std_multiplier: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)
        if footprint_mode not in ("point", "area"):
            raise ValueError(f"unsupported MSI footprint mode: {footprint_mode}")
        self.footprint_mode = footprint_mode
        self.gaussian_enabled = bool(gaussian_enabled)
        self.gaussian_alpha = float(gaussian_alpha)
        if self.gaussian_alpha < 0:
            raise ValueError("gaussian_alpha must be non-negative")

        # Keep registration order through fc2 identical to frozen E6. The new
        # transport has the same head parameter shapes as E6's HR renderer.
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
        self.gaussian_transport = LRCellGaussianTransportDualSource(
            dim, std_multiplier=std_multiplier
        )
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        # Match frozen E6's constructor-level RNG contract: every new-only
        # branch is invisible to the RNG state from which Train_Cave.py starts
        # its global Xavier pass.
        extra_rng_state = torch.get_rng_state()
        self.msi_footprint_sampler = ScaleSharedMSIFootprintSampler()
        self.msi_footprint_embed = nn.Sequential(
            nn.Conv2d(16 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)
        torch.set_rng_state(extra_rng_state)
        self._last_stats: Dict[str, float] = {}
        self.arch_summary = (
            "E6 common HR reconstruction path; one circular primitive per LR-HSI "
            "cell; scale-shared MSI footprint; LR routing; density-normalized "
            "adaptive 3-sigma CUDA transport directly to the HR target grid"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        self.routing.reset_output_init()
        self.gaussian_transport.reset_custom_init()

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
    ):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        h_lr = self.shallow_encoder1(lr_hsi)
        for layer in self.adci_hsi_layers:
            h_lr = layer(h_lr)
        f_m = self.shallow_encoder2(hr_msi)
        for layer in self.adci_msi_layers:
            f_m = layer(f_m)

        f_h = F.interpolate(
            h_lr, size=target_size, mode="bicubic", align_corners=False
        )
        joint_hr = torch.cat((f_h, f_m), dim=1)
        f_fused = self.conv0(joint_hr)

        footprint = self.msi_footprint_sampler(
            f_m, h_lr.shape[-2:], mode=self.footprint_mode
        )
        m_lr = self.msi_footprint_embed(footprint)
        joint_lr = torch.cat((h_lr, m_lr), dim=1)
        e0_lr = self.primitive_input(joint_lr)
        e_g_lr = e0_lr + self.primitive_residual(e0_lr)
        route_lr = self.routing(h_lr, m_lr)
        value_source_lr = e_g_lr + route_lr

        gaussian_delta_hr, gs_aux = self.gaussian_transport(
            transport_x=e_g_lr,
            value_x=value_source_lr,
            out_size=target_size,
            return_aux=True,
        )
        effective_gaussian_alpha = (
            self.gaussian_alpha if self.gaussian_enabled else 0.0
        )
        applied_gaussian_delta_hr = gaussian_delta_hr * effective_gaussian_alpha
        refined = f_fused + applied_gaussian_delta_hr
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            e_g_abs = e_g_lr.detach().abs().mean()
            route_abs = route_lr.detach().abs().mean()
            f_fused_abs = f_fused.detach().abs().mean()
            gaussian_abs = gaussian_delta_hr.detach().abs().mean()
            applied_gaussian_abs = applied_gaussian_delta_hr.detach().abs().mean()
            stats = {
                "footprint_mode_area": float(self.footprint_mode == "area"),
                "gaussian_enabled": float(self.gaussian_enabled),
                "gaussian_alpha": effective_gaussian_alpha,
                "primitive_embedding_abs_mean": float(e_g_abs),
                "routing_delta_abs_mean": float(route_abs),
                "routing_to_embedding_ratio": float(
                    route_abs / (e_g_abs + 1e-8)
                ),
                "f_fused_abs_mean": float(f_fused_abs),
                "gaussian_delta_abs_mean": float(gaussian_abs),
                "gaussian_to_f_fused_ratio": float(
                    gaussian_abs / (f_fused_abs + 1e-8)
                ),
                "applied_gaussian_to_f_fused_ratio": float(
                    applied_gaussian_abs / (f_fused_abs + 1e-8)
                ),
            }
            if self.routing.last_stats:
                stats.update(self.routing.last_stats)
            self._last_stats = stats

        if not return_aux:
            return prediction
        return prediction, {
            "base": base,
            "H_lr": h_lr,
            "F_H": f_h,
            "F_M": f_m,
            "M_lr": m_lr,
            "E_g_lr": e_g_lr,
            "route_lr": route_lr,
            "gaussian_delta_hr": gaussian_delta_hr,
            "applied_gaussian_delta_hr": applied_gaussian_delta_hr,
            "F_fused": f_fused,
            "refined": refined,
            "residual": residual,
            **gs_aux,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        return [] if not stats else [{"layer": "lr_primitive_transport", **stats}]


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
