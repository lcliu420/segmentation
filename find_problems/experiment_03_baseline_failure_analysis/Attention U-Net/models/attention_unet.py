from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate for filtering encoder skip features."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_transform = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_transform = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(self.gate_transform(gate) + self.skip_transform(skip)))
        return skip * attention


class AttentionUNet(nn.Module):
    """2D Attention U-Net for binary gastric lesion segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError("AttentionUNet expects four encoder feature levels.")

        f1, f2, f3, f4 = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(in_channels, f1)
        self.enc2 = DoubleConv(f1, f2)
        self.enc3 = DoubleConv(f2, f3)
        self.enc4 = DoubleConv(f3, f4)
        self.bottleneck = DoubleConv(f4, f4 * 2)

        self.up4 = nn.ConvTranspose2d(f4 * 2, f4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=f4, skip_channels=f4, inter_channels=f4 // 2)
        self.dec4 = DoubleConv(f4 * 2, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=f3, skip_channels=f3, inter_channels=f3 // 2)
        self.dec3 = DoubleConv(f3 * 2, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=f2, skip_channels=f2, inter_channels=f2 // 2)
        self.dec2 = DoubleConv(f2 * 2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=f1, skip_channels=f1, inter_channels=max(f1 // 2, 1))
        self.dec1 = DoubleConv(f1 * 2, f1)

        self.final_conv = nn.Conv2d(f1, num_classes, kernel_size=1)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.up4(bottleneck)
        enc4_att = self.att4(dec4, enc4)
        dec4 = self.dec4(torch.cat([enc4_att, self._match_size(dec4, enc4_att)], dim=1))

        dec3 = self.up3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = self.dec3(torch.cat([enc3_att, self._match_size(dec3, enc3_att)], dim=1))

        dec2 = self.up2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = self.dec2(torch.cat([enc2_att, self._match_size(dec2, enc2_att)], dim=1))

        dec1 = self.up1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = self.dec1(torch.cat([enc1_att, self._match_size(dec1, enc1_att)], dim=1))

        return self.final_conv(dec1)
