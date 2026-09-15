"""
GSFusion v2 for HSI-MSI fusion.

Key ideas:
1. Two shallow branches for HSI and MSI features.
2. A stacked Gaussian-splatting encoder with FFN residual blocks.
3. A low-rank spectral basis at the output stage.
4. Direct 31-band decoder (NoBasis). The old spectral_basis decoder was removed
   from the main v2 path because it did not improve the current experiments.
5. Zero-initialized decoder tail so epoch 0 matches bicubic.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.support.EDSR import make_edsr_baseline
from tools.Utils import make_coord


# ============================================================================
# CUDA rasterizer extension loader
# ============================================================================
def _resolve_gaussian_rasterizer():
    try:
        from diff_srgaussian_rasterization import GaussianRasterizer
        return GaussianRasterizer
    except Exception:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        submodule_root = os.path.join(repo_root, "submodules", "diff-srgaussian-rasterization")
        build_root = os.path.join(submodule_root, "build")

        if os.path.isdir(build_root):
            entries = [e for e in sorted(os.listdir(build_root)) if e.startswith("lib.")]
            py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
            preferred = [e for e in entries if py_tag in e]
            fallback = [e for e in entries if py_tag not in e]
            ordered = preferred + fallback

            for entry in reversed(ordered):
                candidate = os.path.join(build_root, entry)
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)

        if submodule_root not in sys.path:
            sys.path.append(submodule_root)

        from diff_srgaussian_rasterization import GaussianRasterizer
        return GaussianRasterizer


# ============================================================================
# Basic ResBlock used in shallow feature extraction
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class GaussianPrimaryHead(nn.Module):
    def __init__(self, in_features, feat_dim_out):
        super().__init__()
        self.act = nn.ReLU()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            self.act,
            nn.Linear(in_features // 2, 1),
            nn.Sigmoid(),
        )
        self.mlp_rho = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            self.act,
            nn.Linear(in_features // 2, 1),
            nn.Tanh(),
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            self.act,
            nn.Linear(in_features // 2, 2),
        )
        self.mlp_std = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            self.act,
            nn.Linear(in_features // 2, 2),
            nn.Sigmoid(),
        )
        self.mlp_color = nn.Sequential(
            nn.Linear(in_features, in_features),
            self.act,
            nn.Linear(in_features, feat_dim_out),
        )

    def forward(self, gauss_embeds, ref_pos):
        """
        gauss_embeds: (B, C, H, W) pixel-wise features
        ref_pos:      (B, H*W, 2) normalized coordinates in [-1, 1]
        """
        B, C, H, W = gauss_embeds.shape
        gauss_embeds = gauss_embeds.permute(0, 2, 3, 1).reshape(B, H * W, C)

        opacity = self.mlp_opacity(gauss_embeds)
        rho = self.mlp_rho(gauss_embeds) * 0.9999
        offset = 0.1 * torch.tanh(self.mlp_offset(gauss_embeds))
        std = 0.05 + 0.25 * self.mlp_std(gauss_embeds) + 1e-6
        mean = (offset + ref_pos).clamp(-1.0, 1.0)
        color = self.mlp_color(gauss_embeds)
        return opacity, rho, mean, std, color


class GSEncoder(nn.Module):
    """
    Single Gaussian refinement block.
    Keeps the internal channel width equal to dim.
    """

    def __init__(self, dim):
        super().__init__()
        GaussianRasterizer = _resolve_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim)
        self.head = GaussianPrimaryHead(dim, dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

        # Diagnostic cache. This is not a parameter or buffer,
        # so old checkpoints remain compatible.
        self.last_stats = None

    def forward(self, x):
        B, C, H, W = x.shape
        coord_yx = make_coord((H, W), flatten=True).to(x.device)
        ref_pos = torch.stack([coord_yx[:, 1], coord_yx[:, 0]], dim=-1)
        ref_pos = ref_pos.unsqueeze(0).expand(B, H * W, 2)

        opacity, rho, mean, std, color = self.head(x, ref_pos)

        # The CUDA rasterizer expects means/stds in pixel coordinates.
        sx = (W - 1) * 0.5
        sy = (H - 1) * 0.5
        means_px = torch.stack([(mean[..., 0] + 1) * sx, (mean[..., 1] + 1) * sy], dim=-1)
        stds_px = torch.stack([std[..., 0] * sx, std[..., 1] * sy], dim=-1)

        rendered = self.rasterizer(
            opacity.float(),
            means_px.float(),
            stds_px.float(),
            rho.float(),
            color.float(),
            H, W, 1, 0.1, debug=False,
        )
        rendered = rendered.permute(0, 3, 1, 2)

        # Keep the exact same math as:
        # out = x + rendered
        # out = out + self.ffn(out)
        out_gs = x + rendered
        ffn_out = self.ffn(out_gs)
        out = out_gs + ffn_out

        # Diagnostics only. Do not affect gradients or output.
        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            rendered_abs = rendered.detach().abs().mean()
            ffn_abs = ffn_out.detach().abs().mean()

            self.last_stats = {
                "opacity_mean": opacity.detach().mean().item(),
                "opacity_std": opacity.detach().std().item(),
                "opacity_min": opacity.detach().min().item(),
                "opacity_max": opacity.detach().max().item(),
                "opacity_gt_09": (opacity.detach() > 0.9).float().mean().item(),
                "opacity_lt_01": (opacity.detach() < 0.1).float().mean().item(),

                "std_x_mean": std[..., 0].detach().mean().item(),
                "std_y_mean": std[..., 1].detach().mean().item(),
                "std_x_std": std[..., 0].detach().std().item(),
                "std_y_std": std[..., 1].detach().std().item(),

                "rho_mean": rho.detach().mean().item(),
                "rho_abs_mean": rho.detach().abs().mean().item(),

                # mean = ref_pos + offset after clamping.
                # This measures the effective displacement after clamp.
                "effective_offset_abs_mean": (mean.detach() - ref_pos).abs().mean().item(),

                "x_abs_mean": x_abs.item(),
                "rendered_abs_mean": rendered_abs.item(),
                "rendered_x_ratio": (rendered_abs / (x_abs + 1e-8)).item(),

                "ffn_abs_mean": ffn_abs.item(),
                "ffn_x_ratio": (ffn_abs / (x_abs + 1e-8)).item(),
            }

        return out


class GSFusion(nn.Module):
    def __init__(
        self,
        dim=32,
        num_bands=31,
        num_msi=3,
        num_basis=16,
        num_gs_layers=3,
        edsr_resblocks=6,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_basis = num_basis
        self.dim = dim

        # (1) Two shallow feature branches
        self.shallow_hsi = nn.Sequential(
            nn.Conv2d(num_bands, dim, 1),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.shallow_msi = nn.Sequential(
            nn.Conv2d(num_msi, dim, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        # (2) Joint EDSR backbone
        self.edsr_encoder = make_edsr_baseline(
            n_resblocks=edsr_resblocks, n_feats=dim, n_colors=2 * dim
        )

        # (3) Stacked GSEncoder blocks
        self.gs_layers = nn.Sequential(
            *[GSEncoder(dim) for _ in range(num_gs_layers)]
        )

        # (4) Decoder: direct 31-band residual output.
        # num_basis is kept in the constructor only for CLI/checkpoint compatibility.
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_bands, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def collect_gs_stats(self):
        stats = []
        for i, layer in enumerate(self.gs_layers):
            layer_stats = getattr(layer, "last_stats", None)
            if layer_stats is None:
                continue
            item = {"layer": i}
            item.update(layer_stats)
            stats.append(item)
        return stats

    def forward(self, lr_hsi, hr_msi, sf):
        """
        lr_hsi: (B, 31, h, w)
        hr_msi: (B, 3, H, W) where H = h * sf
        """
        # Bicubic upsampling as the residual baseline
        lr_hsi_up = F.interpolate(
            lr_hsi, scale_factor=sf, mode="bicubic", align_corners=False
        )

        f_hsi = self.shallow_hsi(lr_hsi_up)  # (B, dim, H, W)
        f_msi = self.shallow_msi(hr_msi)     # (B, dim, H, W)

        feat = torch.cat([f_hsi, f_msi], dim=1)  # (B, 2*dim, H, W)
        feat = self.edsr_encoder(feat)            # (B, dim, H, W)
        feat = self.gs_layers(feat)               # (B, dim, H, W)

        residual = self.decoder(feat)             # (B, num_bands, H, W)

        return residual + lr_hsi_up


# ============================================================================
# Loss: L1 + SAM with warmup
# ============================================================================
def sam_loss(pred, gt, eps=1e-8):
    """
    1 - cosine similarity as a stable SAM-style loss.
    pred, gt: (B, C, H, W)
    """
    cos = (pred * gt).sum(dim=1) / (
        pred.norm(dim=1) * gt.norm(dim=1) + eps
    )
    return (1.0 - cos).mean()


def compute_loss(pred, gt, epoch, sam_warmup_epochs=5, sam_weight=0.1):
    """
    Use only L1 for the first few epochs, then warm up the SAM term.
    """
    l1 = F.l1_loss(pred, gt)
    if epoch < sam_warmup_epochs:
        return l1
    w = min(sam_weight, sam_weight * (epoch - sam_warmup_epochs + 1) / 5.0)
    return l1 + w * sam_loss(pred, gt)
