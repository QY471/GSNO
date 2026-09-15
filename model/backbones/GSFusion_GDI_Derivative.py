"""E3-GDI-Derivative-v2 controlled enhancement.

The formal GDI-v1 model and its exact GaussianDifferenceIntegralBlock are
reused directly.  This file only wraps each GDI-v1 block with a zero-starting
fixed-HR analytic Gaussian derivative residual (dx, dy, LoG at the same three
scales).  The E3 fusion, primitive embedding, circular normalized Gaussian
transport, spectral head, and raw bicubic residual path are inherited without
change.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.GSFusion_GDI import (
    GSFusion as GDIV1Model,
    GaussianDifferenceIntegralBlock,
    LayerNorm2d,
    _gradient_l2,
)
from model.GSFusion_GSNO import compute_loss, sam_loss


def _normalize_zero_sum_kernel(
    kernel: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    kernel = kernel - kernel.mean()
    return kernel / kernel.abs().sum().clamp_min(eps)


def build_gaussian_derivative_kernels(
    dilation: int,
    sigma: float,
) -> Dict[str, object]:
    if int(dilation) <= 0:
        raise ValueError("dilation must be positive")
    if float(sigma) <= 0:
        raise ValueError("sigma must be positive")

    offsets: List[Tuple[int, int]] = []
    kx_values: List[float] = []
    ky_values: List[float] = []
    log_values: List[float] = []
    sigma2 = float(sigma) ** 2
    sigma4 = sigma2 ** 2

    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            delta_y = float(offset_y * int(dilation))
            delta_x = float(offset_x * int(dilation))
            radius2 = delta_x * delta_x + delta_y * delta_y
            gaussian = math.exp(-radius2 / (2.0 * sigma2))
            offsets.append((offset_y, offset_x))
            kx_values.append(-(delta_x / sigma2) * gaussian)
            ky_values.append(-(delta_y / sigma2) * gaussian)
            log_values.append(
                ((radius2 - 2.0 * sigma2) / sigma4) * gaussian
            )

    return {
        "offsets": offsets,
        "kx": _normalize_zero_sum_kernel(
            torch.tensor(kx_values, dtype=torch.float32)
        ),
        "ky": _normalize_zero_sum_kernel(
            torch.tensor(ky_values, dtype=torch.float32)
        ),
        "log": _normalize_zero_sum_kernel(
            torch.tensor(log_values, dtype=torch.float32)
        ),
    }


class GaussianDerivativeBank(nn.Module):
    """Fixed-buffer analytic derivatives on the GDI-v1 HR stencils."""

    def __init__(
        self,
        dim: int,
        dilations: Sequence[int] = (1, 2, 4),
        sigmas: Sequence[float] = (1.0, 2.0, 4.0),
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        if len(dilations) != len(sigmas):
            raise ValueError("dilations and sigmas must have the same length")
        if not dilations:
            raise ValueError("at least one scale is required")
        if padding_mode not in ("reflect", "replicate"):
            raise ValueError("padding_mode must be reflect or replicate")

        self.dim = int(dim)
        self.dilations = tuple(int(value) for value in dilations)
        self.sigmas = tuple(float(value) for value in sigmas)
        self.padding_mode = padding_mode
        self.base_offsets: Tuple[Tuple[int, int], ...] = tuple(
            (offset_y, offset_x)
            for offset_y in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
        )

        for scale_index, (dilation, sigma) in enumerate(
            zip(self.dilations, self.sigmas)
        ):
            kernels = build_gaussian_derivative_kernels(dilation, sigma)
            if kernels["offsets"] != list(self.base_offsets):
                raise RuntimeError("offset order mismatch with GDI-v1")
            for name in ("kx", "ky", "log"):
                self.register_buffer(
                    f"{name}_{scale_index}",
                    kernels[name],
                    persistent=True,
                )

    def _apply_kernel(
        self,
        x: torch.Tensor,
        dilation: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = x.shape
        if self.padding_mode == "reflect" and (
            height <= dilation or width <= dilation
        ):
            raise ValueError("reflect padding requires H/W > dilation")

        padded = F.pad(
            x,
            (dilation, dilation, dilation, dilation),
            mode=self.padding_mode,
        )
        weights = weights.to(device=x.device, dtype=x.dtype)
        output = torch.zeros_like(x)
        for index, (offset_y, offset_x) in enumerate(self.base_offsets):
            y_start = dilation + offset_y * dilation
            x_start = dilation + offset_x * dilation
            shifted = padded[
                :,
                :,
                y_start : y_start + height,
                x_start : x_start + width,
            ]
            output = output + weights[index] * shifted
        return output

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        responses: List[torch.Tensor] = []
        for scale_index, dilation in enumerate(self.dilations):
            responses.extend(
                [
                    self._apply_kernel(
                        x, dilation, getattr(self, f"kx_{scale_index}")
                    ),
                    self._apply_kernel(
                        x, dilation, getattr(self, f"ky_{scale_index}")
                    ),
                    self._apply_kernel(
                        x, dilation, getattr(self, f"log_{scale_index}")
                    ),
                ]
            )
        return responses


class GDIDerivativeBlock(nn.Module):
    """Exact GDI-v1 block plus a zero-starting derivative residual."""

    def __init__(
        self,
        dim: int,
        gdi_base: GaussianDifferenceIntegralBlock,
        use_msi_gate: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(gdi_base, GaussianDifferenceIntegralBlock):
            raise TypeError("gdi_base must be the real GDI-v1 block")
        self.dim = int(dim)
        self.gdi_base = gdi_base
        self.use_msi_gate = bool(use_msi_gate)
        self.dilations = tuple(gdi_base.dilations)
        self.sigmas = tuple(gdi_base.sigmas)
        self.padding_mode = gdi_base.padding_mode

        self.derivative_norm = LayerNorm2d(dim)
        self.derivative_value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.derivative_bank = GaussianDerivativeBank(
            dim=dim,
            dilations=self.dilations,
            sigmas=self.sigmas,
            padding_mode=self.padding_mode,
        )
        response_channels = dim * len(self.dilations) * 3
        self.derivative_fuse = nn.Sequential(
            nn.Conv2d(response_channels, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )
        if self.use_msi_gate:
            self.guide_gate: Optional[nn.Module] = nn.Sequential(
                LayerNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.guide_gate = None
        self.last_stats: Optional[Dict[str, float]] = None
        self.reset_derivative_init()

    def reset_derivative_init(self) -> None:
        last = self.derivative_fuse[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        x: torch.Tensor,
        msi_guide: torch.Tensor,
        return_aux: bool = False,
    ):
        base_result = self.gdi_base(x, return_aux=return_aux)
        if return_aux:
            base_output, base_aux = base_result
        else:
            base_output = base_result
            base_aux = None

        value = self.derivative_value_proj(self.derivative_norm(x))
        responses = self.derivative_bank(value)
        response_tensor = torch.cat(responses, dim=1)
        derivative_delta = self.derivative_fuse(response_tensor)

        if self.guide_gate is not None:
            if msi_guide.shape != x.shape:
                raise ValueError(
                    f"guide shape {tuple(msi_guide.shape)} != "
                    f"x shape {tuple(x.shape)}"
                )
            gate = self.guide_gate(msi_guide)
            derivative_delta = derivative_delta * gate
        else:
            gate = torch.ones_like(x)
        output = base_output + derivative_delta

        if not self.training:
            with torch.no_grad():
                x_abs = x.detach().abs().mean()
                delta_abs = derivative_delta.detach().abs().mean()
                stats: Dict[str, float] = {
                    "derivative_value_abs_mean": float(value.detach().abs().mean()),
                    "derivative_delta_abs_mean": float(delta_abs),
                    "derivative_delta_input_ratio": float(
                        delta_abs / (x_abs + 1e-8)
                    ),
                    "guide_gate_mean": float(gate.detach().mean()),
                    "guide_gate_std": float(gate.detach().std()),
                    "guide_gate_min": float(gate.detach().min()),
                    "guide_gate_max": float(gate.detach().max()),
                }
                for scale_index, dilation in enumerate(self.dilations):
                    response_index = scale_index * 3
                    stats[f"dx_abs_mean_scale_{dilation}"] = float(
                        responses[response_index].detach().abs().mean()
                    )
                    stats[f"dy_abs_mean_scale_{dilation}"] = float(
                        responses[response_index + 1].detach().abs().mean()
                    )
                    stats[f"log_abs_mean_scale_{dilation}"] = float(
                        responses[response_index + 2].detach().abs().mean()
                    )
                self.last_stats = stats

        if not return_aux:
            return output
        return output, {
            "base_aux": base_aux,
            "derivative_value": value,
            "responses": responses,
            "guide_gate": gate,
            "derivative_delta": derivative_delta,
        }


class GSFusion(GDIV1Model):
    """Formal GDI-v1 model with independent derivative wrappers per branch."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        pointwise_blocks: int = 2,
        pointwise_expansion: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            pointwise_blocks=pointwise_blocks,
            pointwise_expansion=pointwise_expansion,
            **kwargs,
        )
        self.hsi_gdi = GDIDerivativeBlock(dim, gdi_base=self.hsi_gdi)
        self.msi_gdi = GDIDerivativeBlock(dim, gdi_base=self.msi_gdi)
        self._last_gradient_stats = {}
        self._gradient_stat_calls = 0
        self.arch_summary = (
            "E3-GDI-Derivative-v2: exact GDI-v1 pointwise lifting and GDI "
            "base; fixed-HR analytic dx/dy/LoG at dilations=(1,2,4), "
            "sigmas=(1,2,4); MSI channel-amplitude gate; zero-starting "
            "derivative residual; formal E3 Gaussian decoder and head"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        # During GDIV1Model.__init__, the blocks have not yet been wrapped.
        if not isinstance(getattr(self, "hsi_gdi", None), GDIDerivativeBlock):
            GDIV1Model.reset_custom_init(self)
            return

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        for wrapper in (self.hsi_gdi, self.msi_gdi):
            base = wrapper.gdi_base
            nn.init.zeros_(base.scale_gate.weight)
            nn.init.zeros_(base.scale_gate.bias)
            nn.init.zeros_(base.gamma)
            wrapper.reset_derivative_init()

    def collect_gradient_stats(self) -> Dict[str, float]:
        self._gradient_stat_calls += 1
        if self._gradient_stat_calls > 2 and self._gradient_stat_calls % 250:
            return dict(self._last_gradient_stats)

        stats: Dict[str, float] = {}
        for name, wrapper in (("hsi", self.hsi_gdi), ("msi", self.msi_gdi)):
            base = wrapper.gdi_base
            stats.update(
                {
                    f"grad_{name}_gdi_gamma": _gradient_l2([base.gamma]),
                    f"grad_{name}_scale_gate": _gradient_l2(
                        base.scale_gate.parameters()
                    ),
                    f"grad_{name}_gdi_value_proj": _gradient_l2(
                        base.value_proj.parameters()
                    ),
                    f"grad_{name}_derivative_fuse_last": _gradient_l2(
                        wrapper.derivative_fuse[-1].parameters()
                    ),
                    f"grad_{name}_derivative_fuse_first": _gradient_l2(
                        wrapper.derivative_fuse[0].parameters()
                    ),
                    f"grad_{name}_derivative_value_proj": _gradient_l2(
                        wrapper.derivative_value_proj.parameters()
                    ),
                    f"grad_{name}_guide_gate": _gradient_l2(
                        wrapper.guide_gate.parameters()
                        if wrapper.guide_gate is not None
                        else []
                    ),
                }
            )
        self._last_gradient_stats = stats
        return dict(stats)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        gaussian_stats = self.gaussian_refine.last_stats
        if gaussian_stats is not None:
            records.append({"layer": "hr_gaussian", **gaussian_stats})
        for branch, wrapper in (
            ("hsi_gdi_derivative", self.hsi_gdi),
            ("msi_gdi_derivative", self.msi_gdi),
        ):
            combined: Dict[str, float] = {"layer": branch}
            if wrapper.gdi_base.last_stats is not None:
                combined.update(wrapper.gdi_base.last_stats)
            if wrapper.last_stats is not None:
                combined.update(wrapper.last_stats)
            if len(combined) > 1:
                combined.update(self._last_gradient_stats)
                records.append(combined)
        return records

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        f_hsi_lr = self.shallow_encoder1(lr_hsi)
        for block in self.hsi_point_blocks:
            f_hsi_lr = block(f_hsi_lr)
        f_hsi_hr = F.interpolate(
            f_hsi_lr, size=target_size, mode="bicubic", align_corners=False
        )

        f_msi = self.shallow_encoder2(hr_msi)
        for block in self.msi_point_blocks:
            f_msi = block(f_msi)
        msi_guide = f_msi

        f_hsi_hr = self.hsi_gdi(
            f_hsi_hr, msi_guide=msi_guide
        )
        f_msi = self.msi_gdi(
            f_msi, msi_guide=msi_guide
        )

        joint = torch.cat((f_hsi_hr, f_msi), dim=1)
        f_fused = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        primitive_with_delta = self.gaussian_refine(e_g)
        gaussian_delta = primitive_with_delta - e_g
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "GDIDerivativeBlock",
    "GaussianDerivativeBank",
    "build_gaussian_derivative_kernels",
    "compute_loss",
    "sam_loss",
]
