"""Self-contained LR-cell Gaussian-difference GSNO.

The model retains the strong GSNO ADCI/bicubic backbone, applies one zero-sum
Gaussian difference whose dilation follows the LR-cell scale, then uses three
pointwise residual FFN blocks. This file intentionally contains every runtime
dependency of CellDiff so the curated model has one obvious implementation entry.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """The channel-last LayerNorm used by the original GSNO ADCI."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalization for branch response stabilization."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class ADCI(nn.Module):
    """Unchanged 3x3 local ADCI from the original strong GSNO."""

    def __init__(self, in_channels: int, mlp_hidden_dim: int):
        super().__init__()
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = torch.chunk(self.qkv_conv(x), chunks=3, dim=1)

        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(b, c, 9, h, w)
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(b, c, 9, h, w)

        q_minus_k = q.unsqueeze(2) - k_unfold
        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()
        attention_scores = F.softmax(self.mlp(q_minus_k), dim=-2)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * attention_scores, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()
        return weighted_v + self.gate(x)


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-0.5 * (xx.square() + yy.square()) / (sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_difference_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Zero-sum Gaussian averaging residual: Gaussian(x) - x."""

    kernel = _gaussian_kernel(kernel_size, sigma)
    center = kernel_size // 2
    kernel[center, center] -= 1.0
    # Remove tiny floating-point drift so constants are preserved exactly enough.
    kernel -= kernel.sum() / float(kernel.numel())
    return kernel


class FixedDepthwiseSpatialKernel(nn.Module):
    """Memory-efficient fixed spatial response with integer-dilation fast path.

    For the formal 4/8/16/32 tests, the LR-cell dilation is exactly 1/2/4/8,
    so the implementation uses one depthwise convolution instead of materializing
    a B x C x K^2 x H x W unfold tensor.
    """

    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("kernel must be a square 2D tensor")
        if kernel.shape[0] % 2 != 1:
            raise ValueError("kernel size must be odd")
        self.kernel_size = int(kernel.shape[0])
        self.radius = self.kernel_size // 2
        self.register_buffer("kernel", kernel.view(1, 1, self.kernel_size, self.kernel_size))

    @staticmethod
    def _is_integer(value: float, atol: float = 1e-6) -> bool:
        return abs(value - round(value)) <= atol

    def _integer_forward(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        channels = x.shape[1]
        pad = self.radius * dilation
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        weight = self.kernel.to(dtype=x.dtype).expand(channels, 1, -1, -1).contiguous()
        return F.conv2d(x_pad, weight, dilation=dilation, groups=channels)

    def _fractional_forward(self, x: torch.Tensor, dilation: float) -> torch.Tensor:
        """Slow fallback for non-integer scales; formal scales use the fast path."""

        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype
        yy = torch.arange(h, device=device, dtype=dtype)
        xx = torch.arange(w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        base_x = (2.0 * (gx + 0.5) / float(w)) - 1.0
        base_y = (2.0 * (gy + 0.5) / float(h)) - 1.0

        kernel = self.kernel[0, 0].to(device=device, dtype=dtype)
        out = torch.zeros_like(x)
        for iy in range(self.kernel_size):
            for ix in range(self.kernel_size):
                coeff = kernel[iy, ix]
                if float(coeff.abs()) < 1e-12:
                    continue
                dy = (iy - self.radius) * dilation
                dx = (ix - self.radius) * dilation
                grid_x = base_x + 2.0 * dx / float(w)
                grid_y = base_y + 2.0 * dy / float(h)
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
                sampled = F.grid_sample(
                    x,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                out = out + coeff * sampled
        return out

    def forward(self, x: torch.Tensor, dilation: float = 1.0) -> torch.Tensor:
        if self._is_integer(float(dilation)):
            return self._integer_forward(x, int(round(float(dilation))))
        return self._fractional_forward(x, float(dilation))


class ResidualFFNBlock(nn.Module):
    """The ordinary channel FFN retained after removing legacy Gaussian gather."""

    def __init__(self, dim: int, expansion: int = 4, zero_init: bool = True):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.ReLU(),
            nn.Conv2d(dim * expansion, dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.ffd[-1].weight)
            nn.init.zeros_(self.ffd[-1].bias)
        self.last_stats: Optional[Dict[str, float]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.ffd(x)
        out = x + residual
        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            residual_abs = residual.detach().abs().mean()
            self.last_stats = {
                "x_abs_mean": float(x_abs),
                "ffn_abs_mean": float(residual_abs),
                "ffn_x_ratio": float(residual_abs / (x_abs + 1e-8)),
            }
        return out


class _BaseGaussianRefinement(nn.Module):
    def __init__(self, dim: int, branch_count: int):
        super().__init__()
        hidden = max(dim, 16)
        self.branch_norms = nn.ModuleList([LayerNorm2d(dim) for _ in range(branch_count)])
        self.merge = nn.Sequential(
            nn.Conv2d(dim * branch_count, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        # Safe start: the complete new Gaussian operator initially equals Identity.
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)
        self.last_stats: Optional[Dict[str, float]] = None

    def _merge_responses(
        self,
        x: torch.Tensor,
        responses: Iterable[torch.Tensor],
        routes: Optional[Iterable[torch.Tensor]] = None,
        extra_stats: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        response_list = list(responses)
        if len(response_list) != len(self.branch_norms):
            raise RuntimeError("response count does not match configured branch count")
        normalized = [norm(resp) for norm, resp in zip(self.branch_norms, response_list)]
        route_list = None if routes is None else list(routes)
        if route_list is not None:
            if len(route_list) != len(normalized):
                raise RuntimeError("route count does not match response count")
            normalized = [route * resp for route, resp in zip(route_list, normalized)]

        merged = torch.cat(normalized, dim=1)
        residual = self.merge(merged)
        out = x + residual

        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            residual_abs = residual.detach().abs().mean()
            stats: Dict[str, float] = {
                "x_abs_mean": float(x_abs),
                "gaussian_residual_abs_mean": float(residual_abs),
                "gaussian_residual_x_ratio": float(residual_abs / (x_abs + 1e-8)),
            }
            for index, response in enumerate(response_list):
                stats[f"response_{index}_abs_mean"] = float(response.detach().abs().mean())
            if extra_stats:
                stats.update(extra_stats)
            self.last_stats = stats
        return out


class CellGaussianDifferenceRefinement(_BaseGaussianRefinement):
    """GPU-0 variant: only the LR-cell transported Gaussian difference."""

    def __init__(
        self,
        dim: int,
        canonical_scale: float = 4.0,
        kernel_size: int = 5,
        sigma_canonical_px: float = 1.0,
    ):
        super().__init__(dim=dim, branch_count=1)
        self.canonical_scale = float(canonical_scale)
        self.cell_kernel = FixedDepthwiseSpatialKernel(
            _gaussian_difference_kernel(kernel_size, sigma_canonical_px)
        )

    def forward(
        self,
        x: torch.Tensor,
        msi_feature: torch.Tensor,
        scale_y: float,
        scale_x: float,
    ) -> torch.Tensor:
        del msi_feature
        if abs(scale_x - scale_y) > 1e-6:
            raise ValueError("Current fast implementation expects isotropic scale")
        dilation = float(scale_x) / self.canonical_scale
        response = self.cell_kernel(x, dilation=dilation)
        return self._merge_responses(
            x,
            [response],
            extra_stats={"cell_dilation": float(dilation)},
        )


REFINEMENT_REGISTRY = {
    "cell": CellGaussianDifferenceRefinement,
}


class GSFusionBase(nn.Module):
    """Strong GSNO backbone with one configurable Gaussian feature operator."""

    def __init__(
        self,
        refinement: str,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        num_basis: int = 16,
        num_gs_layers: int = 3,
        edsr_resblocks: int = 6,
        adci_layers: int = 3,
        canonical_scale: float = 4.0,
        ffn_layers: int = 3,
        ffn_zero_init: bool = True,
        **refinement_kwargs,
    ):
        super().__init__()
        del num_basis, num_gs_layers, edsr_resblocks
        if refinement not in REFINEMENT_REGISTRY:
            raise ValueError(f"Unknown refinement={refinement!r}")

        self.num_bands = num_bands
        self.dim = dim
        self.refinement_name = refinement
        self.canonical_scale = float(canonical_scale)

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])
        self.adci_msi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])

        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        refine_cls = REFINEMENT_REGISTRY[refinement]
        valid_refine_args = set(inspect.signature(refine_cls.__init__).parameters) - {"self"}
        filtered_refine_kwargs = {
            key: value for key, value in refinement_kwargs.items() if key in valid_refine_args
        }
        self.ignored_constructor_kwargs = sorted(
            key for key in refinement_kwargs if key not in valid_refine_args
        )
        self.gaussian_refine = refine_cls(
            dim=dim,
            canonical_scale=canonical_scale,
            **filtered_refine_kwargs,
        )
        self.ffn_layers = nn.ModuleList(
            [ResidualFFNBlock(dim, zero_init=ffn_zero_init) for _ in range(ffn_layers)]
        )

        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.gaussian_refine.merge[-1].weight)
        nn.init.zeros_(self.gaussian_refine.merge[-1].bias)

    def collect_gs_stats(self):
        stats = []
        refine_stats = getattr(self.gaussian_refine, "last_stats", None)
        if refine_stats is not None:
            item = {"module": "gaussian_refine", "variant": self.refinement_name}
            item.update(refine_stats)
            stats.append(item)
        for index, layer in enumerate(self.ffn_layers):
            layer_stats = getattr(layer, "last_stats", None)
            if layer_stats is None:
                continue
            item = {"module": "ffn", "layer": index}
            item.update(layer_stats)
            stats.append(item)
        return stats

    @staticmethod
    def _unwrap_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        for key in ("state_dict", "model", "model_state_dict"):
            nested = state_dict.get(key) if isinstance(state_dict, Mapping) else None
            if isinstance(nested, Mapping):
                return nested
        return state_dict

    def load_legacy_gsno_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        map_old_ffn: bool = True,
    ) -> Dict[str, object]:
        """Load a legacy GSNO/NoGS checkpoint without requiring strict key identity.

        Common backbone parameters are loaded by exact key and shape. When a
        legacy Gaussian GSNO checkpoint is supplied, gs_layers.i.ffd parameters
        can initialize the new ordinary ffn_layers.i.ffd blocks.
        """

        source = self._unwrap_state_dict(state_dict)
        current = self.state_dict()
        mapped: MutableMapping[str, torch.Tensor] = OrderedDict()
        skipped = []

        for raw_key, tensor in source.items():
            if not isinstance(tensor, torch.Tensor):
                skipped.append(str(raw_key))
                continue
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            target_key = key
            if map_old_ffn and key.startswith("gs_layers.") and ".ffd." in key:
                target_key = key.replace("gs_layers.", "ffn_layers.", 1)
            if target_key in current and current[target_key].shape == tensor.shape:
                mapped[target_key] = tensor
            else:
                skipped.append(key)

        result = self.load_state_dict(mapped, strict=False)
        return {
            "loaded_keys": sorted(mapped.keys()),
            "loaded_count": len(mapped),
            "skipped_source_keys": skipped,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf) -> torch.Tensor:
        target_h, target_w = hr_msi.shape[-2:]
        lr_h, lr_w = lr_hsi.shape[-2:]
        scale_y = target_h / float(lr_h)
        scale_x = target_w / float(lr_w)

        # Use target size rather than scale_factor to avoid silent rounding mismatch.
        lr_hsi_up = F.interpolate(
            lr_hsi,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        f_msi = self.shallow_encoder2(hr_msi)
        f_hsi = self.shallow_encoder1(lr_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        f_hsi = F.interpolate(
            f_hsi,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        feat = self.conv0(torch.cat([f_hsi, f_msi], dim=1))
        feat = self.gaussian_refine(feat, f_msi, scale_y=scale_y, scale_x=scale_x)
        for layer in self.ffn_layers:
            feat = layer(feat)

        residual = self.fc2(F.gelu(self.fc1(feat)))
        return residual + lr_hsi_up

    def forward_with_diagnostics(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf,
    ):
        """Run an unchanged prediction and expose validation-only mechanism probes.

        The ADCI residual is measured in the common fused latent space by
        comparing features produced with and without the ADCI stacks. The
        Gaussian residual is the refinement output minus its fused input, so
        both tensors have identical shape and their cosine is well-defined.
        """

        del sf
        target_h, target_w = hr_msi.shape[-2:]
        lr_h, lr_w = lr_hsi.shape[-2:]
        scale_y = target_h / float(lr_h)
        scale_x = target_w / float(lr_w)

        lr_hsi_up = F.interpolate(
            lr_hsi,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        f_msi_initial = self.shallow_encoder2(hr_msi)
        f_hsi_initial = self.shallow_encoder1(lr_hsi)
        f_msi = f_msi_initial
        f_hsi = f_hsi_initial
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        f_hsi_initial_up = F.interpolate(
            f_hsi_initial,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        f_hsi = F.interpolate(
            f_hsi,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        feat_without_adci = self.conv0(
            torch.cat([f_hsi_initial_up, f_msi_initial], dim=1)
        )
        feat_before_gaussian = self.conv0(torch.cat([f_hsi, f_msi], dim=1))
        feat_after_gaussian = self.gaussian_refine(
            feat_before_gaussian,
            f_msi,
            scale_y=scale_y,
            scale_x=scale_x,
        )

        adci_residual = feat_before_gaussian - feat_without_adci
        gaussian_residual = feat_after_gaussian - feat_before_gaussian
        adci_flat = adci_residual.flatten(1)
        gaussian_flat = gaussian_residual.flatten(1)
        cosine = F.cosine_similarity(adci_flat, gaussian_flat, dim=1, eps=1e-8)
        adci_abs = adci_residual.detach().abs().mean()
        gaussian_abs = gaussian_residual.detach().abs().mean()
        diagnostics: Dict[str, float] = {
            "adci_gaussian_cosine": float(cosine.detach().mean()),
            "adci_residual_abs_mean": float(adci_abs),
            "gaussian_residual_abs_mean": float(gaussian_abs),
            "gaussian_to_adci_abs_ratio": float(gaussian_abs / (adci_abs + 1e-8)),
            "scale_y": float(scale_y),
            "scale_x": float(scale_x),
        }
        refine_stats = getattr(self.gaussian_refine, "last_stats", None)
        if refine_stats:
            diagnostics.update(refine_stats)

        feat = feat_after_gaussian
        for layer in self.ffn_layers:
            feat = layer(feat)
        residual = self.fc2(F.gelu(self.fc1(feat)))
        return residual + lr_hsi_up, diagnostics


class GSFusion(GSFusionBase):
    """Formal CellDiff model used by Train_Cave.py."""

    def __init__(self, **kwargs):
        super().__init__(refinement="cell", **kwargs)


def sam_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cos = (pred * gt).sum(dim=1) / (pred.norm(dim=1) * gt.norm(dim=1) + eps)
    return (1.0 - cos).mean()


def compute_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    epoch: int,
    sam_warmup_epochs: int = 5,
    sam_weight: float = 0.1,
) -> torch.Tensor:
    l1 = F.l1_loss(pred, gt)
    if epoch < sam_warmup_epochs:
        return l1
    weight = min(sam_weight, sam_weight * (epoch - sam_warmup_epochs + 1) / 5.0)
    return l1 + weight * sam_loss(pred, gt)
