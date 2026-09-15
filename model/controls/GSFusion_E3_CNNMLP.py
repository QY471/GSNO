"""Joint CNN + pointwise MLP control, preserving the E3 two-stream topology.

ADCI is replaced by three ordinary 3x3-GELU-3x3 residual blocks per stream.
The primitive/Gaussian branch is replaced by a capacity-matched 1x1 MLP.
No ADCI or Gaussian module is instantiated or executed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model.GSFusion_GSNO import compute_loss, sam_loss


class CNNResidual(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, dim, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.gelu(self.conv1(x)))


class GSFusion(nn.Module):
    def __init__(self, dim=80, num_bands=31, num_msi=3, adci_layers=3, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.num_msi = num_msi
        # Parameter formulas are independently checked against the actual
        # formal DIM80 model by the preflight audit, not inferred from results.
        self.adci_block_target = 6 * dim * dim + 5 * dim
        self.gaussian_branch_target = 7 * dim * dim + 10 * dim + 4
        self.cnn_hidden = min(range(1, 4097), key=lambda h: abs(h*(18*dim+1)+dim-self.adci_block_target))
        self.mlp_hidden = min(range(1, 4097), key=lambda h: abs(h*(3*dim+1)+dim-self.gaussian_branch_target))
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.cnn_hsi_layers = nn.ModuleList([CNNResidual(dim, self.cnn_hidden) for _ in range(adci_layers)])
        self.cnn_msi_layers = nn.ModuleList([CNNResidual(dim, self.cnn_hidden) for _ in range(adci_layers)])
        self.conv0 = nn.Sequential(nn.Conv2d(2*dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.pointwise_latent = nn.Sequential(nn.Conv2d(2*dim, self.mlp_hidden, 1), nn.GELU(), nn.Conv2d(self.mlp_hidden, dim, 1))
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        self.arch_summary = 'E3 CNN+MLP joint ablation: two native-grid streams, 3 ordinary residual CNN blocks each; matched pointwise MLP replaces primitive+Gaussian; same fusion, decoder and bicubic base; no ADCI, no Gaussian'
        self.reset_custom_init()

    def reset_custom_init(self):
        for layer in list(self.cnn_hsi_layers) + list(self.cnn_msi_layers):
            nn.init.zeros_(layer.conv2.weight)
            nn.init.zeros_(layer.conv2.bias)
        nn.init.zeros_(self.pointwise_latent[-1].weight)
        nn.init.zeros_(self.pointwise_latent[-1].bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, lr_hsi, hr_msi, sf=None, return_aux=False, output_size=None):
        reference = hr_msi.shape[-2:]
        query = reference if output_size is None else tuple(output_size)
        base = F.interpolate(lr_hsi, size=query, mode='bicubic', align_corners=False)
        hsi, msi = self.shallow_encoder1(lr_hsi), self.shallow_encoder2(hr_msi)
        for layer in self.cnn_hsi_layers:
            hsi = layer(hsi)
        for layer in self.cnn_msi_layers:
            msi = layer(msi)
        hsi = F.interpolate(hsi, size=reference, mode='bicubic', align_corners=False)
        joint = torch.cat((hsi, msi), dim=1)
        fused, delta = self.conv0(joint), self.pointwise_latent(joint)
        refined = fused + delta
        if tuple(query) != tuple(reference):
            refined = F.interpolate(refined, size=query, mode='bicubic', align_corners=False)
        prediction = base + self.fc2(F.gelu(self.fc1(refined)))
        if return_aux:
            return prediction, {'F_H': hsi, 'F_M': msi, 'F_fused_reference': fused, 'pointwise_delta_reference': delta}
        return prediction
