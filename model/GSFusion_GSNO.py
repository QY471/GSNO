"""
GSFusion GSNO Adapter - connect the original GSNO model to train.py.

Minimal changes from the original GSNO:
1. Rename the trainable class to GSFusion.
2. Add collect_gs_stats() for diagnostics.
3. Add compute_loss / sam_loss.
4. Accept standard train.py kwargs such as dim, num_bands, num_msi.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        out = (x - mean) / (std + self.eps)
        return self.weight * out + self.bias


class ADCI(nn.Module):
    def __init__(self, in_channels, mlp_hidden_dim):
        super().__init__()
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(b, c, 9, h, w)
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(b, c, 9, h, w)

        q_expanded = q.unsqueeze(2)
        q_minus_k = q_expanded - k_unfold

        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()
        mlp_output = self.mlp(q_minus_k)
        attention_scores = F.softmax(mlp_output, dim=-2)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * attention_scores, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()

        return weighted_v + self.gate(x)


class GaussianSplatEncoder(nn.Module):
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

        self.mlp_params = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.sx_param = nn.Parameter(torch.ones(1) * 0.5)
        self.sy_param = nn.Parameter(torch.ones(1) * 0.5)
        self.rho_param = nn.Parameter(torch.zeros(1))

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

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        feat = x.view(B, C, N).permute(0, 2, 1)
        color_flat = self.mlp_color(feat)
        params_flat = self.mlp_params(feat)

        raw_sx, raw_sy, raw_rho = params_flat.unbind(-1)

        sx_map = self.sx_param * raw_sx.view(B, 1, H, W)
        sy_map = self.sy_param * raw_sy.view(B, 1, H, W)
        rho_map = self.rho_param * raw_rho.view(B, 1, H, W)

        color_map = color_flat.permute(0, 2, 1).view(B, -1, H, W)
        col_unf = F.unfold(color_map, self.K, padding=self.r)
        sx_unf = F.unfold(sx_map, self.K, padding=self.r)
        sy_unf = F.unfold(sy_map, self.K, padding=self.r)
        rho_unf = F.unfold(rho_map, self.K, padding=self.r)

        D = col_unf.shape[1] // (self.K * self.K)
        col_unf = col_unf.view(B, D, self.K * self.K, N).permute(0, 3, 2, 1)
        sx = torch.sigmoid(sx_unf.view(B, self.K * self.K, N)) + self.eps
        sy = torch.sigmoid(sy_unf.view(B, self.K * self.K, N)) + self.eps
        rho = torch.tanh(rho_unf.view(B, self.K * self.K, N))

        det = (sx * sy) ** 2 * (1 - rho ** 2) + self.eps
        inv_f = 1.0 / (1 - rho ** 2 + self.eps)
        inv11 = inv_f / (sx ** 2 + self.eps)
        inv22 = inv_f / (sy ** 2 + self.eps)
        inv12 = -rho * inv_f / (sx * sy + self.eps)

        offsets = self.offsets.to(x.device)
        dy = offsets[:, 0].view(1, self.K * self.K, 1)
        dx = offsets[:, 1].view(1, self.K * self.K, 1)

        d = inv11 * dy ** 2 + inv22 * dx ** 2 + 2 * inv12 * dx * dy

        denom = 2 * math.pi * det
        w = torch.exp(-0.5 * d) / (denom + self.eps)
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)

        out_flat = torch.einsum("bkn,bnkd->bnd", w, col_unf)
        out = out_flat.permute(0, 2, 1).view(B, D, H, W)

        out = out + x
        ffd_out = self.ffd(out)
        out = ffd_out + out

        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            ffd_abs = ffd_out.detach().abs().mean()
            self.last_stats = {
                "sx_param": self.sx_param.detach().item(),
                "sy_param": self.sy_param.detach().item(),
                "rho_param": self.rho_param.detach().item(),
                "sx_mean": sx.detach().mean().item(),
                "sy_mean": sy.detach().mean().item(),
                "rho_mean": rho.detach().mean().item(),
                "rho_abs_mean": rho.detach().abs().mean().item(),
                "x_abs_mean": x_abs.item(),
                "ffd_abs_mean": ffd_abs.item(),
                "ffd_x_ratio": (ffd_abs / (x_abs + 1e-8)).item(),
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
    ):
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

        self.gs_layers = nn.ModuleList(
            [
                GaussianSplatEncoder(in_channels=dim, color_dim=dim, hidden_dim=dim)
                for _ in range(num_gs_layers)
            ]
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
            layer_stats = getattr(layer, "last_stats", None)
            if layer_stats is None:
                continue
            item = {"layer": i}
            item.update(layer_stats)
            stats.append(item)
        return stats

    def forward(self, lr_hsi, hr_msi, sf):
        lr_hsi_up = F.interpolate(
            lr_hsi, scale_factor=sf, mode="bicubic", align_corners=False
        )

        f_msi = self.shallow_encoder2(hr_msi)
        f_hsi = self.shallow_encoder1(lr_hsi)

        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        f_hsi = F.interpolate(f_hsi, scale_factor=sf, mode="bicubic", align_corners=False)

        feat = torch.cat([f_hsi, f_msi], dim=1)
        feat = self.conv0(feat)

        for layer in self.gs_layers:
            feat = layer(feat)

        ret = self.fc2(F.gelu(self.fc1(feat)))
        return ret + lr_hsi_up


def sam_loss(pred, gt, eps=1e-8):
    cos = (pred * gt).sum(dim=1) / (pred.norm(dim=1) * gt.norm(dim=1) + eps)
    return (1.0 - cos).mean()


def compute_loss(pred, gt, epoch, sam_warmup_epochs=5, sam_weight=0.1):
    l1 = F.l1_loss(pred, gt)
    if epoch < sam_warmup_epochs:
        return l1
    w = min(sam_weight, sam_weight * (epoch - sam_warmup_epochs + 1) / 5.0)
    return l1 + w * sam_loss(pred, gt)
