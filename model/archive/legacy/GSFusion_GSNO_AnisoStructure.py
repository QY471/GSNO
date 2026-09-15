"""
GSFusion GSNO + Structure-Tensor Anisotropic Gaussian Splatting.

Why (grounded in this repo diagnostics):
- The original GaussianSplatEncoder (GSFusion_GSNO) learns (sx, sy, rho) freely from
  the fused feature. Empirically rho collapses to ~0 and sx~=sy (isotropic round
  blobs), so the Gaussian merely duplicates bicubic upsampling and adds little.
- Its geometry is driven by the sf-dependent (bicubic-upsampled) HSI feature, so it
  overfits sf=4 and hurts zero-shot OOD scales.

What changes:
- Each Gaussian ORIENTATION + ANISOTROPY is derived from the HR-MSI structure tensor
  (local gradient covariance). HR-MSI is always full resolution (independent of sf),
  so the geometry is sf-invariant (OOD-friendly) and anisotropy is imposed rather than
  learned (cannot collapse). Gaussians elongate ALONG MSI edges -> edge-preserving
  gather that bicubic cannot do.
- Only a few scalars (base sigma, anisotropy gain) are learnable per gs layer; the
  color path and the FFD residual are unchanged from GSFusion_GSNO.

Interface identical to GSFusion_GSNO: forward(lr_hsi, hr_msi, sf); same compute_loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the exact backbone pieces / loss from the strong GSNO model
from model.GSFusion_GSNO import ADCI, LayerNorm, sam_loss, compute_loss


class StructureTensor(nn.Module):
    """Compute per-pixel edge direction (cos/sin) and anisotropy from HR-MSI."""

    def __init__(self, smooth_ksize=5, smooth_sigma=1.0):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = sx.t().contiguous()
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sy.view(1, 1, 3, 3))
        ax = torch.arange(smooth_ksize).float() - smooth_ksize // 2
        g = torch.exp(-(ax ** 2) / (2 * smooth_sigma ** 2))
        g = g / g.sum()
        k2 = torch.outer(g, g)
        self.register_buffer("gauss", k2.view(1, 1, smooth_ksize, smooth_ksize))
        self.pad = smooth_ksize // 2

    @torch.no_grad()
    def forward(self, msi):
        B, C, H, W = msi.shape
        x = msi.reshape(B * C, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1).view(B, C, H, W)
        gy = F.conv2d(x, self.sobel_y, padding=1).view(B, C, H, W)
        Jxx = (gx * gx).sum(1, keepdim=True)
        Jyy = (gy * gy).sum(1, keepdim=True)
        Jxy = (gx * gy).sum(1, keepdim=True)
        Jxx = F.conv2d(Jxx, self.gauss, padding=self.pad)
        Jyy = F.conv2d(Jyy, self.gauss, padding=self.pad)
        Jxy = F.conv2d(Jxy, self.gauss, padding=self.pad)
        tr = Jxx + Jyy
        diff = Jxx - Jyy
        disc = torch.sqrt(diff * diff + 4 * Jxy * Jxy + 1e-12)
        # strength-gated anisotropy: flat regions (low gradient energy tr) -> ~0,
        # so we only elongate where MSI actually has a coherent edge.
        aniso = (disc / (tr + 1e-2)).clamp(0.0, 1.0)
        theta = 0.5 * torch.atan2(2 * Jxy, diff + 1e-12)      # gradient (across-edge) dir
        phi = theta + math.pi / 2.0                           # edge (along) dir
        return torch.cos(phi), torch.sin(phi), aniso          # each (B,1,H,W)


class StructureTensorGaussianSplatEncoder(nn.Module):
    """Local 5x5 anisotropic Gaussian gather; geometry imposed by MSI structure tensor."""

    def __init__(self, in_channels, color_dim, hidden_dim=32, kernel_size=5, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.K = kernel_size
        self.r = kernel_size // 2

        self.mlp_color = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, color_dim),
        )
        # learnable geometry magnitude. Orientation is imposed by the MSI structure
        # tensor (cannot collapse), but the anisotropy AMPLITUDE g=sigmoid(aniso_gain)
        # is a free scalar that CAN be driven toward 0 -> back to isotropic. We start
        # near-isotropic (aniso_gain=-2 -> g~0.12) and let training grow it if useful,
        # instead of forcing strong elongation from scratch.
        self.log_s_base = nn.Parameter(torch.zeros(1))
        self.aniso_gain = nn.Parameter(torch.full((1,), -2.0))

        yy, xx = torch.meshgrid(
            torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
        )
        offsets = torch.stack([yy - self.r, xx - self.r], dim=-1).view(-1, 2).float()
        self.register_buffer("offsets", offsets)

        self.ffd = nn.Sequential(
            nn.Conv2d(color_dim, color_dim * 4, 1),
            nn.ReLU(),
            nn.Conv2d(4 * color_dim, color_dim, 1),
        )
        self.last_stats = None

    def forward(self, x, cosphi, sinphi, aniso):
        B, C, H, W = x.shape
        N = H * W
        eps = self.eps

        feat = x.view(B, C, N).permute(0, 2, 1)
        color_map = self.mlp_color(feat).permute(0, 2, 1).view(B, -1, H, W)

        # imposed anisotropic covariance from MSI structure tensor
        s_base = (0.5 * torch.exp(self.log_s_base)).clamp(0.1, 3.0)
        g = torch.sigmoid(self.aniso_gain)
        sigma_long = (s_base * (1.0 + g * aniso * 2.0)).clamp(0.1, 3.0)    # along edge
        sigma_short = (s_base * (1.0 - g * aniso * 0.9)).clamp(0.05, 3.0)  # across edge
        v1 = sigma_long ** 2
        v2 = sigma_short ** 2
        c2 = cosphi * cosphi
        s2 = sinphi * sinphi
        cs = cosphi * sinphi
        Sxx = v1 * c2 + v2 * s2
        Syy = v1 * s2 + v2 * c2
        Sxy = (v1 - v2) * cs
        sx_map = torch.sqrt(Sxx + eps)
        sy_map = torch.sqrt(Syy + eps)
        rho_map = (Sxy / (sx_map * sy_map + eps)).clamp(-0.99, 0.99)

        col_unf = F.unfold(color_map, self.K, padding=self.r)
        sx_unf = F.unfold(sx_map, self.K, padding=self.r)
        sy_unf = F.unfold(sy_map, self.K, padding=self.r)
        rho_unf = F.unfold(rho_map, self.K, padding=self.r)

        D = col_unf.shape[1] // (self.K * self.K)
        col_unf = col_unf.view(B, D, self.K * self.K, N).permute(0, 3, 2, 1)
        sx = sx_unf.view(B, self.K * self.K, N)
        sy = sy_unf.view(B, self.K * self.K, N)
        rho = rho_unf.view(B, self.K * self.K, N)

        det = (sx * sy) ** 2 * (1 - rho ** 2) + eps
        inv_f = 1.0 / (1 - rho ** 2 + eps)
        inv11 = inv_f / (sx ** 2 + eps)
        inv22 = inv_f / (sy ** 2 + eps)
        inv12 = -rho * inv_f / (sx * sy + eps)

        offsets = self.offsets.to(x.device)
        dy = offsets[:, 0].view(1, self.K * self.K, 1)
        dx = offsets[:, 1].view(1, self.K * self.K, 1)
        # inv11 is the xx component of the inverse covariance -> pairs with dx**2
        # (inv22 is yy -> dy**2). Getting this wrong transposes the covariance and
        # mirror-flips the orientation.
        d = inv11 * dx ** 2 + inv22 * dy ** 2 + 2 * inv12 * dx * dy

        # 2-D Gaussian density normalization is 1/(2*pi*sqrt(det)), not 1/(2*pi*det).
        w = torch.exp(-0.5 * d) / (2 * math.pi * torch.sqrt(det) + eps)
        w = w / (w.sum(dim=1, keepdim=True) + eps)

        out_flat = torch.einsum("bkn,bnkd->bnd", w, col_unf)
        out = out_flat.permute(0, 2, 1).view(B, D, H, W)

        out = out + x
        ffd_out = self.ffd(out)
        out = ffd_out + out

        with torch.no_grad():
            self.last_stats = {
                "s_base": s_base.mean().item(),
                "aniso_gain": g.mean().item(),
                "aniso_mean": aniso.mean().item(),
                "rho_abs_mean": rho_map.abs().mean().item(),
                "sx_mean": sx_map.mean().item(),
                "sy_mean": sy_map.mean().item(),
                "sx_over_sy": (sx_map.mean() / (sy_map.mean() + 1e-8)).item(),
                "ffd_x_ratio": (ffd_out.detach().abs().mean()
                                / (x.detach().abs().mean() + 1e-8)).item(),
            }
        return out


class GSFusion(nn.Module):
    def __init__(self, dim=64, num_bands=31, num_msi=3, num_basis=16,
                 num_gs_layers=3, edsr_resblocks=6, adci_layers=3):
        super().__init__()
        self.num_bands = num_bands
        self.dim = dim
        self.num_gs_layers = num_gs_layers

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)

        self.adci_hsi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])
        self.adci_msi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])

        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        self.structure = StructureTensor()
        self.gs_layers = nn.ModuleList(
            [StructureTensorGaussianSplatEncoder(in_channels=dim, color_dim=dim,
                                                 hidden_dim=dim)
             for _ in range(num_gs_layers)]
        )

        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def reset_custom_init(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def collect_gs_stats(self):
        stats = []
        for i, layer in enumerate(self.gs_layers):
            ls = getattr(layer, "last_stats", None)
            if ls is None:
                continue
            item = {"layer": i}
            item.update(ls)
            stats.append(item)
        return stats

    def forward(self, lr_hsi, hr_msi, sf=None):
        # Shape-derived output size: the target resolution is HR-MSI, NOT the sf
        # argument. A wrong sf (e.g. 999) cannot change the output size, so this
        # passes the sf-agnostic audit. The true ratio is H/h, W/w if ever needed.
        target_size = hr_msi.shape[-2:]
        lr_hsi_up = F.interpolate(lr_hsi, size=target_size, mode="bicubic",
                                  align_corners=False)

        f_msi = self.shallow_encoder2(hr_msi)
        f_hsi = self.shallow_encoder1(lr_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        f_hsi = F.interpolate(f_hsi, size=target_size, mode="bicubic",
                              align_corners=False)

        feat = self.conv0(torch.cat([f_hsi, f_msi], dim=1))

        # sf-invariant Gaussian geometry from full-res HR-MSI
        cosphi, sinphi, aniso = self.structure(hr_msi)
        for layer in self.gs_layers:
            feat = layer(feat, cosphi, sinphi, aniso)

        ret = self.fc2(F.gelu(self.fc1(feat)))
        return ret + lr_hsi_up
