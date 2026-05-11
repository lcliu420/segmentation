from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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


class UNetPlusPlus(nn.Module):
    """2D U-Net++ with nested dense skip pathways.

    This follows the topology of the official Keras UNetPlusPlus implementation
    in keras/helper_functions.py, adapted to PyTorch and two-class logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        features: tuple[int, ...] = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if len(features) != 5:
            raise ValueError("UNetPlusPlus expects five feature levels.")
        self.deep_supervision = deep_supervision
        nb = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])

        self.up1_0 = nn.ConvTranspose2d(nb[1], nb[0], kernel_size=2, stride=2)
        self.conv0_1 = ConvBlock(nb[0] * 2, nb[0])

        self.up2_0 = nn.ConvTranspose2d(nb[2], nb[1], kernel_size=2, stride=2)
        self.conv1_1 = ConvBlock(nb[1] * 2, nb[1])

        self.up1_1 = nn.ConvTranspose2d(nb[1], nb[0], kernel_size=2, stride=2)
        self.conv0_2 = ConvBlock(nb[0] * 3, nb[0])

        self.up3_0 = nn.ConvTranspose2d(nb[3], nb[2], kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock(nb[2] * 2, nb[2])

        self.up2_1 = nn.ConvTranspose2d(nb[2], nb[1], kernel_size=2, stride=2)
        self.conv1_2 = ConvBlock(nb[1] * 3, nb[1])

        self.up1_2 = nn.ConvTranspose2d(nb[1], nb[0], kernel_size=2, stride=2)
        self.conv0_3 = ConvBlock(nb[0] * 4, nb[0])

        self.up4_0 = nn.ConvTranspose2d(nb[4], nb[3], kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(nb[3] * 2, nb[3])

        self.up3_1 = nn.ConvTranspose2d(nb[3], nb[2], kernel_size=2, stride=2)
        self.conv2_2 = ConvBlock(nb[2] * 3, nb[2])

        self.up2_2 = nn.ConvTranspose2d(nb[2], nb[1], kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock(nb[1] * 4, nb[1])

        self.up1_3 = nn.ConvTranspose2d(nb[1], nb[0], kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock(nb[0] * 5, nb[0])

        self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._match_size(self.up1_0(x1_0), x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._match_size(self.up2_0(x2_0), x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._match_size(self.up1_1(x1_1), x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._match_size(self.up3_0(x3_0), x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._match_size(self.up2_1(x2_1), x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._match_size(self.up1_2(x1_2), x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._match_size(self.up4_0(x4_0), x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._match_size(self.up3_1(x3_1), x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._match_size(self.up2_2(x2_2), x1_0)], dim=1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._match_size(self.up1_3(x1_3), x0_0)], dim=1)
        )

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        return self.final4(x0_4)
