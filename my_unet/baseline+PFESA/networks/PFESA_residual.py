import numpy as np
import torch
import torch.nn as nn


class PFESA(nn.Module):
    """
    Residual PFESA variant.

    It keeps the original PFESA frequency attention logic, but starts as an
    identity skip with a learnable residual strength.
    """
    def __init__(self, base_ratio=0.1):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.base_ratio = base_ratio
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.zeros(1))

    def _edge_attention(self, x):
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        x_var = x.var(dim=[2, 3], keepdim=True)
        return x_minus_mu_square / (x_var + self.eps)

    def _structure_attention(self, x):
        energy_low = x.pow(2)
        energy_mu = energy_low.mean(dim=[2, 3], keepdim=True)
        energy_var = energy_low.var(dim=[2, 3], keepdim=True)
        return self.activation((energy_low - energy_mu) / (energy_var + self.eps))

    def forward(self, x):
        _, _, h, w = x.size()

        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

        low_freq_mask = self._create_low_freq_mask(h, w, device=x_freq.device)
        high_freq_mask = 1 - low_freq_mask

        low_freq = torch.abs(torch.fft.ifftn(x_freq * low_freq_mask, dim=(-2, -1)))
        high_freq = torch.abs(torch.fft.ifftn(x_freq * high_freq_mask, dim=(-2, -1)))

        out_att = self._structure_attention(low_freq) + self._edge_attention(high_freq)
        out_att = self.activation(out_att)
        return x + self.gamma * out_att * x

    def _create_low_freq_mask(self, h, w, device="cpu"):
        mask_ratio = self.base_ratio * min(h, w) / max(h, w)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return torch.exp(-(y_grid ** 2 + x_grid ** 2) / (2 * mask_ratio ** 2))


class TokenPFESA(nn.Module):
    def __init__(self, base_ratio=0.1):
        super().__init__()
        self.pfesa = PFESA(base_ratio=base_ratio)

    def forward(self, x):
        b, length, channels = x.shape
        h = w = int(np.sqrt(length))
        if h * w != length:
            raise ValueError(f"PFESA skip feature expects square tokens, got length={length}.")
        x_img = x.transpose(1, 2).contiguous().view(b, channels, h, w)
        x_img = self.pfesa(x_img)
        return x_img.flatten(2).transpose(1, 2).contiguous()
