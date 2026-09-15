import os
import sys
import math
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.support.EDSR import make_edsr_baseline

from tools.Utils import make_coord


def _resolve_gaussian_rasterizer():
    try:
        from diff_srgaussian_rasterization import GaussianRasterizer
        return GaussianRasterizer
    except Exception:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        submodule_root = os.path.join(repo_root, "submodules", "diff-srgaussian-rasterization")
        build_root = os.path.join(submodule_root, "build")
        build_lib_dirs = []
        if os.path.isdir(build_root):
            build_lib_dirs = sorted(
                os.path.join(build_root, name)
                for name in os.listdir(build_root)
                if name.startswith("lib.")
            )
        for path in reversed(build_lib_dirs + [submodule_root]):
            if path not in sys.path:
                sys.path.insert(0, path)
        from diff_srgaussian_rasterization import GaussianRasterizer
        return GaussianRasterizer

class GsFusion(nn.Module):
    def __init__(self, dim=32, num_gs_layers=3, use_gaussian=True):
        super().__init__()

        self.shallow_encoder = nn.Conv2d(34, dim, 1)
        self.act = nn.ReLU()
        self.use_gaussian = use_gaussian
        self.num_gs_layers = max(int(num_gs_layers), 0)

        if self.use_gaussian and self.num_gs_layers > 0:
            self.encoder = nn.Sequential(*[GSEncoder(dim) for _ in range(self.num_gs_layers)])
        else:
            self.encoder = nn.Identity()

        self.edsr_encoder = make_edsr_baseline(n_resblocks=6, n_feats=dim, n_colors=34)

        self.decoder = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim, 31, kernel_size=1)
        )
        

       


    def forward(self, lr_hsi, hr_msi, sf):
        lr_hsi_up = F.interpolate(lr_hsi, scale_factor=sf, mode='bicubic', align_corners=False)
        hm_si = torch.cat([hr_msi, lr_hsi_up], dim=1)

        hm_si = self.edsr_encoder(hm_si)
        feat = self.encoder(hm_si)
        ret = self.decoder(feat) + lr_hsi_up
        return ret
    

class GSEncoder(nn.Module):
    def __init__(self,dim_in):
        super().__init__()
        GaussianRasterizer = _resolve_gaussian_rasterizer()
        self.gaussian_rasterizer = GaussianRasterizer(dim_in)
        self.gaussian_primary_head = GaussianPrimaryHead(dim_in, dim_in)


    
    def forward(self, x):
        B,C,H,W = x.shape
        coord_yx = make_coord((H, W), flatten=True).to(x.device)
        coord_xy = torch.stack([coord_yx[:, 1], coord_yx[:, 0]], dim=-1)
        coord = coord_xy.unsqueeze(0).expand(B, H * W, 2)
        opacity, rho, mean, std, color = self.gaussian_primary_head(x, coord)

        # Normalized [-1, 1] -> pixel coordinates [0, size-1]
        sx = (W - 1) * 0.5
        sy = (H - 1) * 0.5
        means_px = torch.stack([(mean[..., 0] + 1) * sx, (mean[..., 1] + 1) * sy], dim=-1)
        stds_px = torch.stack([std[..., 0] * sx, std[..., 1] * sy], dim=-1)

        opacity_fp32 = opacity.float()
        rhos_fp32 = rho.float()
        means_fp32 = means_px.float()
        stds_fp32 = stds_px.float()
        colors_fp32 = color.float()
        output_image = self.gaussian_rasterizer(opacity_fp32, means_fp32, stds_fp32, rhos_fp32, colors_fp32, H, W, 1, 0.1, debug=False)

        
        output_image = output_image.permute(0, 3, 1, 2) + x

        return output_image



class GaussianPrimaryHead(nn.Module):
    def __init__(self, in_features, num_colors):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()


        feat_opacity_rho = 1
        feat_offs_std = 2
        feat_color = num_colors

        self.mlp_opacity = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            self.act_fn,
            nn.Linear(in_features//2, feat_opacity_rho),
            self.sigmoid
        )
        self.mlp_rho = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            self.act_fn,
            nn.Linear(in_features//2, feat_opacity_rho),
            self.tanh,
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            self.act_fn,
            nn.Linear(in_features//2, feat_offs_std),
        )
        self.mlp_std = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            self.act_fn,
            nn.Linear(in_features//2, feat_offs_std),
            self.sigmoid
        )
        self.mlp_color = nn.Sequential(
            nn.Linear(in_features, in_features),
            self.act_fn,
            nn.Linear(in_features, feat_color),
        )
    
    def forward(self, gauss_embeds, ref_pos):
        B, C, H, W = gauss_embeds.shape
        gauss_embeds = gauss_embeds.permute(0,2,3,1).reshape(B,H*W,C)

        opacity = self.mlp_opacity(gauss_embeds)
        rho = self.mlp_rho(gauss_embeds)
        offset = 0.1 * torch.tanh(self.mlp_offset(gauss_embeds))
        std = 0.05 + 0.25 * self.mlp_std(gauss_embeds)
        color = self.mlp_color(gauss_embeds) #* self.window_size

        # Safety mechanisms to prevent edge cases
        std = std + 1e-6
        rho = rho * 0.9999

        mean = (offset + ref_pos).clamp(-1.0, 1.0)
        return opacity, rho, mean, std, color
