"""
GSNO-style backbone with MSI-guided HSI Gaussian Upsampling.

Main idea:
  1. Keep 1x1 point-wise HSI/MSI projection to preserve discrete feature representation.
  2. Replace bicubic HSI feature upsampling with MSI-guided 2D Gaussian splatting.
  3. Use PixelUnshuffle / space-to-depth to preserve HR-MSI sub-pixel details for geometry prediction.
  4. Optional Local Cross-ADCI after HSI Gaussian upsampling.
  5. Optional v2 GSEncoder refinement. Default is False to avoid duplicated Gaussian splatting.
  6. Decode with 1x1 fc1/fc2 and bicubic residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI
from model.important_model_support.GSFusionv2 import (
    GSEncoder,
    compute_loss,
    sam_loss,
    _resolve_gaussian_rasterizer,
)


class MSIGuidedHSIGaussianUpsampler(nn.Module):
    """
    MSI-guided Gaussian upsampler for LR-HSI features.

    Input:
        f_hsi_lr: [B, C, h, w]
        f_msi_hr: [B, C, H, W]

    Output:
        f_hsi_hr: [B, C, H, W]

    Design:
        - color/content comes from HSI feature.
        - geometry parameters come from HSI feature + MSI sub-pixel guidance.
        - HR-MSI guidance uses a shared normalized 4x4 footprint sampler.
    """

    def __init__(
        self,
        dim,
        std_min_cell=0.125,
        std_max_cell=1.0,
        max_offset_cell=0.5,
        alpha_init=-2.0,
    ):
        super().__init__()
        GaussianRasterizer = _resolve_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)

        self.dim = dim
        self.std_min_cell = std_min_cell
        self.std_max_cell = std_max_cell
        self.max_offset_cell = max_offset_cell
        # Non-parameter diagnostic controls. Defaults preserve the formal 3-sigma model.
        self.raster_sigma_radius = 3.0
        self.raster_ratio_override = None

        # HSI feature supplies Gaussian color / latent content.
        self.color_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Shared normalized 4x4 footprint guidance for every scale.
        offs = [-0.375, -0.125, 0.125, 0.375]
        grid = torch.tensor([(oy, ox) for oy in offs for ox in offs])
        self.register_buffer("foot_offsets", grid, persistent=False)
        self.msi_foot_embed = nn.Sequential(
            nn.Conv2d(dim * 16, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        # HSI + MSI guidance predict geometry.
        # Output:
        #   opacity: 1
        #   offset_xy_px: 2
        #   std_xy_px: 2
        #   rho: 1
        # total = 6
        self.param_head = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, 6, kernel_size=1),
        )

        # Blend between bicubic HSI feature and Gaussian-rendered HSI feature.
        # Normalized rendering is amplitude-stable, so Gaussian features can
        # start with a meaningful blend weight.
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))

        self.last_stats = None

    def _msi_footprint_guidance(self, f_msi_hr, h, w):
        """Sample 16 normalized positions inside every LR footprint."""
        B, C, H, W = f_msi_hr.shape
        dev, dt = f_msi_hr.device, f_msi_hr.dtype

        # Numerically exact implementation of the same 16 footprint points at
        # the training scale. All scales share the single msi_foot_embed below.
        if H == 4 * h and W == 4 * w:
            pts = F.pixel_unshuffle(f_msi_hr, downscale_factor=4)
            return self.msi_foot_embed(pts)

        cy = torch.arange(h, device=dev, dtype=dt) + 0.5
        cx = torch.arange(w, device=dev, dtype=dt) + 0.5
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")

        offs = self.foot_offsets.to(dt)
        py = gy.unsqueeze(0) + offs[:, 0].view(16, 1, 1)
        px = gx.unsqueeze(0) + offs[:, 1].view(16, 1, 1)
        gyn = 2.0 * py / h - 1.0
        gxn = 2.0 * px / w - 1.0
        grid = torch.stack([gxn, gyn], dim=-1)
        grid = grid.view(1, 16 * h, w, 2).expand(B, -1, -1, -1)

        pts = F.grid_sample(
            f_msi_hr,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        pts = pts.view(B, C, 16, h, w).reshape(B, C * 16, h, w)
        return self.msi_foot_embed(pts)

    def _make_base_pixel_coords(self, h, w, H, W, device, dtype):
        """
        Make LR cell centers mapped to HR pixel coordinates.
        This follows align_corners=False style center alignment:
            x_hr = (x_lr + 0.5) * W / w - 0.5
            y_hr = (y_lr + 0.5) * H / h - 0.5
        """
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (H / h) - 0.5
        x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (W / w) - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack([xx, yy], dim=-1).view(1, h * w, 2)
        return base

    def forward(self, f_hsi_lr, f_msi_hr, target_size):
        B, C, h, w = f_hsi_lr.shape
        H, W = target_size

        # Stable baseline feature upsampling.
        bicubic = F.interpolate(
            f_hsi_lr,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        f_msi_lr = self._msi_footprint_guidance(f_msi_hr, h, w)
        guidance_mode = "shared_footprint_4x4"

        param_in = torch.cat([f_hsi_lr, f_msi_lr], dim=1)
        raw = self.param_head(param_in)  # [B, 6, h, w]
        raw = raw.permute(0, 2, 3, 1).contiguous().view(B, h * w, 6)

        raw_opacity = raw[..., 0:1]
        raw_offset = raw[..., 1:3]
        raw_std = raw[..., 3:5]
        raw_rho = raw[..., 5:6]

        opacity = 0.05 + 0.95 * torch.sigmoid(raw_opacity)

        sy, sx = H / h, W / w
        offset_cell = self.max_offset_cell * torch.tanh(raw_offset)
        std_cell = self.std_min_cell + (
            self.std_max_cell - self.std_min_cell
        ) * torch.sigmoid(raw_std)
        scale_vec = raw.new_tensor([sx, sy])
        offset_px = offset_cell * scale_vec
        std_px = std_cell * scale_vec
        rho = 0.999 * torch.tanh(raw_rho)

        color = self.color_proj(f_hsi_lr)
        color = color.permute(0, 2, 3, 1).contiguous().view(B, h * w, C)

        base_px = self._make_base_pixel_coords(
            h=h,
            w=w,
            H=H,
            W=W,
            device=f_hsi_lr.device,
            dtype=f_hsi_lr.dtype,
        ).expand(B, h * w, 2)

        means_px = base_px + offset_px
        means_px_x = means_px[..., 0].clamp(0.0, float(W - 1))
        means_px_y = means_px[..., 1].clamp(0.0, float(H - 1))
        means_px = torch.stack([means_px_x, means_px_y], dim=-1)

        # The CUDA window is raster_ratio * image size. Anchor it to three
        # standard deviations in LR-cell units so crop size and sf cannot
        # change the physical truncation radius.
        if self.raster_ratio_override is None:
            raster_ratio = min(
                1.0,
                max(
                    self.raster_sigma_radius * self.std_max_cell / w,
                    self.raster_sigma_radius * self.std_max_cell / h,
                ),
            )
        else:
            raster_ratio = float(self.raster_ratio_override)
        ones = color.new_ones(B, h * w, 1)
        vals = torch.cat([color, ones], dim=-1)
        rendered = self.rasterizer(
            opacity.float(),
            means_px.float(),
            std_px.float(),
            rho.float(),
            vals.float(),
            H,
            W,
            1,
            raster_ratio,
            debug=False,
        )
        rendered = rendered.permute(0, 3, 1, 2).contiguous()
        num, den = rendered[:, :C], rendered[:, C:]
        rendered = num / den.clamp_min(1e-6)

        alpha = torch.sigmoid(self.alpha_raw)

        # Blend, not direct addition:
        # alpha small -> close to bicubic
        # alpha large -> trust Gaussian-rendered HSI feature more
        out = (1.0 - alpha) * bicubic + alpha * rendered

        with torch.no_grad():
            bicubic_abs = bicubic.detach().abs().mean()
            rendered_abs = rendered.detach().abs().mean()
            out_abs = out.detach().abs().mean()

            self.last_stats = {
                "hsi_gs_alpha": alpha.detach().item(),
                "hsi_gs_guidance_mode": guidance_mode,
                "hsi_gs_opacity_mean": opacity.detach().mean().item(),
                "hsi_gs_opacity_std": opacity.detach().std().item(),
                "hsi_gs_std_x_mean_cell": std_cell[..., 0].detach().mean().item(),
                "hsi_gs_std_y_mean_cell": std_cell[..., 1].detach().mean().item(),
                "hsi_gs_std_x_min_cell": std_cell[..., 0].detach().min().item(),
                "hsi_gs_std_x_max_cell": std_cell[..., 0].detach().max().item(),
                "hsi_gs_rho_abs_mean": rho.detach().abs().mean().item(),
                "hsi_gs_offset_abs_mean_cell": offset_cell.detach().abs().mean().item(),
                "hsi_gs_offset_max_abs_cell": offset_cell.detach().abs().max().item(),
                "hsi_gs_offset_abs_mean_px": offset_px.detach().abs().mean().item(),
                "hsi_gs_std_x_mean_px": std_px[..., 0].detach().mean().item(),
                "hsi_gs_std_y_mean_px": std_px[..., 1].detach().mean().item(),
                "hsi_gs_std_x_min_px": std_px[..., 0].detach().min().item(),
                "hsi_gs_std_x_max_px": std_px[..., 0].detach().max().item(),
                "hsi_gs_offset_max_abs_px": offset_px.detach().abs().max().item(),
                "hsi_gs_raster_ratio": raster_ratio,
                "hsi_gs_den_min": den.detach().min().item(),
                "hsi_gs_den_mean": den.detach().mean().item(),
                "hsi_gs_bicubic_abs_mean": bicubic_abs.item(),
                "hsi_gs_rendered_abs_mean": rendered_abs.item(),
                "hsi_gs_rendered_bicubic_ratio": (
                    rendered_abs / (bicubic_abs + 1e-8)
                ).item(),
                "hsi_gs_alpha_rendered_bicubic_ratio": (
                    alpha * rendered_abs / (bicubic_abs + 1e-8)
                ).item(),
                "hsi_gs_out_abs_mean": out_abs.item(),
            }

        return out


class LocalCrossADCI(nn.Module):
    """
    Gated local cross-modal ADCI.

    HSI is Query.
    MSI is Key/Value.

    Injects local MSI spatial guidance into HSI features without static 3x3 conv.
    """

    def __init__(self, dim, mlp_hidden_dim=None, kernel_size=3, gamma_init=0.0):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = dim

        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_neighbors = kernel_size * kernel_size

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # gamma=0 starts exactly as identity.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self.last_stats = None

    def forward(self, hsi, msi):
        if hsi.shape != msi.shape:
            raise ValueError(
                f"LocalCrossADCI expects same shape, got hsi={hsi.shape}, msi={msi.shape}"
            )

        B, C, H, W = hsi.shape

        q = self.q_proj(hsi)
        k = self.k_proj(msi)
        v = self.v_proj(msi)

        k_unfold = F.unfold(
            k,
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).view(B, C, self.num_neighbors, H, W)

        v_unfold = F.unfold(
            v,
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).view(B, C, self.num_neighbors, H, W)

        q_minus_k = q.unsqueeze(2) - k_unfold

        # [B, H, W, K*K, C]
        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()

        scores = self.mlp(q_minus_k)

        # Softmax over local neighbors.
        attn = F.softmax(scores, dim=3)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()

        cross = torch.sum(neighbors_v * attn, dim=3)
        cross = cross.permute(0, 3, 1, 2).contiguous()
        cross = self.out_proj(cross)

        out = hsi + self.gamma * cross

        with torch.no_grad():
            hsi_abs = hsi.detach().abs().mean()
            cross_abs = cross.detach().abs().mean()
            self.last_stats = {
                "cross_gamma": self.gamma.detach().item(),
                "cross_abs_mean": cross_abs.item(),
                "hsi_abs_mean": hsi_abs.item(),
                "cross_hsi_ratio": (cross_abs / (hsi_abs + 1e-8)).item(),
            }

        return out


class GSFusion(nn.Module):
    def __init__(
        self,
        dim=64,
        num_bands=31,
        num_msi=3,
        num_basis=16,
        num_gs_layers=3,
        edsr_resblocks=6,
        adci_layers=3,
        use_cross_adci=True,
        use_v2_refine=False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.dim = dim
        self.num_gs_layers = num_gs_layers
        self.num_basis = num_basis
        self.edsr_resblocks = edsr_resblocks
        self.use_cross_adci = use_cross_adci
        self.use_v2_refine = use_v2_refine

        self.arch_summary = (
            f"GSNOBackbone+MSIGuidedHSIGaussianUpsample"
            f"+{'CrossADCI+' if use_cross_adci else ''}"
            f"{'v2ScatterRefine+' if use_v2_refine else ''}"
            f"Decoder: dim={dim}, ADCI={adci_layers} per stream, "
            f"HSI_up=MSI-guided Gaussian splat with PixelUnshuffle guidance, "
            f"fusion=conv0 1x1 stack, decoder=fc1+GELU+fc2"
        )

        # Point-wise projection only. No static 3x3 encoder.
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, kernel_size=1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, kernel_size=1)

        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )

        # Replace bicubic HSI feature upsampling.
        self.hsi_gs_upsampler = MSIGuidedHSIGaussianUpsampler(dim)

        if use_cross_adci:
            self.cross_adci = LocalCrossADCI(dim, dim)

        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        if use_v2_refine:
            self.gs_layers = nn.ModuleList(
                [GSEncoder(dim) for _ in range(num_gs_layers)]
            )
        else:
            self.gs_layers = nn.ModuleList()

        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc2 = nn.Conv2d(dim, num_bands, kernel_size=1)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def reset_custom_init(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def collect_gs_stats(self):
        stats = []

        up_stats = getattr(self.hsi_gs_upsampler, "last_stats", None)
        if up_stats is not None:
            item = {"layer": "hsi_gs_upsampler"}
            item.update(up_stats)
            stats.append(item)

        if self.use_cross_adci:
            cross_stats = getattr(self.cross_adci, "last_stats", None)
            if cross_stats is not None:
                item = {"layer": "cross_adci"}
                item.update(cross_stats)
                stats.append(item)

        if self.use_v2_refine:
            for i, layer in enumerate(self.gs_layers):
                layer_stats = getattr(layer, "last_stats", None)
                if layer_stats is None:
                    continue
                item = {"layer": i}
                item.update(layer_stats)
                stats.append(item)

        return stats

    def forward(self, lr_hsi, hr_msi, sf):
        target_size = hr_msi.shape[-2:]

        # Final residual base.
        lr_hsi_up = F.interpolate(
            lr_hsi,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)

        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        # New core module:
        # LR-HSI feature -> MSI-guided Gaussian splatting -> HR-HSI feature.
        f_hsi = self.hsi_gs_upsampler(
            f_hsi_lr=f_hsi,
            f_msi_hr=f_msi,
            target_size=target_size,
        )

        # Optional cross-modal local guidance at HR.
        if self.use_cross_adci:
            f_hsi = self.cross_adci(f_hsi, f_msi)

        feat = torch.cat([f_hsi, f_msi], dim=1)
        feat = self.conv0(feat)

        # Optional. Default False to avoid duplicated Gaussian splatting.
        if self.use_v2_refine:
            for layer in self.gs_layers:
                feat = layer(feat)

        residual = self.fc2(F.gelu(self.fc1(feat)))

        return lr_hsi_up + residual
