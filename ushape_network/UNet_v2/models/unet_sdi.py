import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SDISkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projections = nn.ModuleList([
            ConvBNReLU(channel, out_channels, kernel_size=1)
            for channel in in_channels
        ])
        self.smooths = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features, target):
        target_size = target.shape[-2:]
        fused = None
        for feature, projection, smooth in zip(features, self.projections, self.smooths):
            x = projection(feature)
            if x.shape[-2:] != target_size:
                if x.shape[-2] > target_size[0] or x.shape[-1] > target_size[1]:
                    x = F.adaptive_avg_pool2d(x, target_size)
                else:
                    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            x = smooth(x)
            fused = x if fused is None else fused * x
        return fused


class UpBlock(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSDI(nn.Module):
    def __init__(self, n_classes=1, input_channels=3, channels=(32, 64, 128, 256, 512), **kwargs):
        super().__init__()
        self.deep_supervision = False
        self.n_classes = n_classes
        self.channels = channels

        c0, c1, c2, c3, c4 = channels
        self.pool = nn.MaxPool2d(2, 2)

        self.enc0 = DoubleConv(input_channels, c0)
        self.enc1 = DoubleConv(c0, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)

        encoder_channels = [c0, c1, c2, c3, c4]
        self.sdi3 = SDISkip(encoder_channels, c3)
        self.sdi2 = SDISkip(encoder_channels, c2)
        self.sdi1 = SDISkip(encoder_channels, c1)
        self.sdi0 = SDISkip(encoder_channels, c0)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, c0, c0)

        self.final = nn.Conv2d(c0, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        features = [x0, x1, x2, x3, x4]

        skip3 = self.sdi3(features, x3)
        d3 = self.up3(x4, skip3)

        skip2 = self.sdi2(features, x2)
        d2 = self.up2(d3, skip2)

        skip1 = self.sdi1(features, x1)
        d1 = self.up1(d2, skip1)

        skip0 = self.sdi0(features, x0)
        d0 = self.up0(d1, skip0)

        return self.final(d0)
