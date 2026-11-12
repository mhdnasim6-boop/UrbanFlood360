"""
UrbanFlood360 U-Net v2 - written for this project (not repository code).

Classic U-Net (Ronneberger et al., 2015) for per-pixel flood segmentation of
the 6-channel Sentinel-1 stack:

    encoder:    4 x [DoubleConv -> MaxPool(2)]       32 -> 64 -> 128 -> 256
    bottleneck: DoubleConv + Dropout                 512
    decoder:    4 x [ConvTranspose(2) -> concat skip -> DoubleConv]
    head:       1x1 conv -> 1 logit per pixel

Differences from the repository Unet in models/unet/model.py:
  - BatchNorm after every 3x3 conv (stabilises training on a small dataset)
  - Dropout only in the bottleneck instead of inside every block
  - returns raw logits (sigmoid applied by the caller), so the loss can use the
    numerically stable BCEWithLogits with a pos_weight for the flood class
Input height/width must be divisible by 16 (four 2x poolings).
"""
import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.MaxPool2d(2), DoubleConv(c_in, c_out))


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_out * 2, c_out)

    def forward(self, x, skip):
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base=32, dropout=0.3):
        super().__init__()
        c = [base * 2**i for i in range(5)]  # 32, 64, 128, 256, 512
        self.inc = DoubleConv(in_channels, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4])
        self.drop = nn.Dropout2d(dropout)
        self.up4 = Up(c[4], c[3])
        self.up3 = Up(c[3], c[2])
        self.up2 = Up(c[2], c[1])
        self.up1 = Up(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.drop(self.down4(x4))
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
