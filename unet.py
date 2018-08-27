####################################
# Yet another unet implementation.
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

class NetG_32(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, nz=512):
        super(NetG_32, self).__init__()

        # UNET style Autoencoder layers.
        self.inc = InpConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 512)
        # self.bneck = FeatConv(512, nz)
        self.up1 = UpConv(1024, 256)
        self.up2 = UpConv(512, 128)
        self.up3 = OutConv(128, 64)
        self.up4 = OutConv(64, 64)
        # self.up3 = UpConv(256, 64)    # TODO This is OutConv
        # self.up4 = UpConv(128, 64)    # TODO This is OutConv
        self.outc = OutConv(64, n_classes)

        # Feature Encoder layers.
        self.enc0 = InpConv(n_channels, 64)
        self.enc1 = DownConv(64, 128)
        self.enc2 = DownConv(128, 256)
        self.enc3 = DownConv(256, 512)
        self.enc4 = DownConv(512, 512)
        self.bneck_ = FeatConv(512, nz)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # z = self.bneck(x5)
        xu1 = self.up1(x5, x4)
        xu2 = self.up2(xu1, x3)
        xu3 = self.up3(xu2) # TODO This is OutConv
        xu4 = self.up4(xu3) # TODO This is OutConv
        # xu3 = self.up3(xu2, x2) # TODO This is OutConv
        # xu4 = self.up4(xu3, x1) # TODO This is OutConv
        xu5 = self.outc(xu4)
        z_ = self.enc0(x)
        z_ = self.enc1(z_)
        z_ = self.enc2(z_)
        z_ = self.enc3(z_)
        z_ = self.enc4(z_)
        z_ = self.bneck_(z_)

        # return x, z, z_
        return xu5, xu4, xu3, xu2, xu1, x5, x4, x3, x2, x1

# class NetG_32(nn.Module):
#     def __init__(self, n_channels=3, n_classes=3, nz=512):
#         super(NetG_32, self).__init__()

#         # UNET style Autoencoder layers.
#         self.inc = InpConv(n_channels, 64)
#         self.down1 = DownConv(64, 128)
#         self.down2 = DownConv(128, 256)
#         self.down3 = DownConv(256, 512)
#         self.down4 = DownConv(512, 512)
#         self.bneck = FeatConv(512, nz)
#         self.up1 = UpConv(1024, 256)
#         self.up2 = UpConv(512, 128)
#         self.up3 = UpConv(256, 64)
#         self.up4 = UpConv(128, 64)
#         self.outc = OutConv(64, n_classes)

#         # Feature Encoder layers.
#         self.enc0 = InpConv(n_channels, 64)
#         self.enc1 = DownConv(64, 128)
#         self.enc2 = DownConv(128, 256)
#         self.enc3 = DownConv(256, 512)
#         self.enc4 = DownConv(512, 512)
#         self.bneck_ = FeatConv(512, nz)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         z = self.bneck(x5)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         z_ = self.enc0(x)
#         z_ = self.enc1(z_)
#         z_ = self.enc2(z_)
#         z_ = self.enc3(z_)
#         z_ = self.enc4(z_)
#         z_ = self.bneck_(z_)

#         return x, z, z_

class InpConv(nn.Module):
    def __init__(self, inp_chn, out_chn, kernel_size=4, stride=2, padding=1):
        super(InpConv, self).__init__()
        self.iconv = nn.Conv2d(inp_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.iconv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, inp_chn, out_chn, kernel_size=4, stride=2, padding=1):
        super(DownConv, self).__init__()
        self.dconv = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inp_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_chn, eps=1e-5, momentum=0.1, affine=True)
        )
    def forward(self, x):
        x = self.dconv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, inp_chn, out_chn, kernel_size=4, stride=2, padding=1, bilinear=False):
        super(UpConv, self).__init__()
        if bilinear:
            self.upconv = nn.Upsample(scale_factor=2, mode='bilinear')
            # self.uconv = nn.Sequential(
            #     nn.ReLU(inplace=True),
            #     nn.Upsample(scale_factor=2, mode='bilinear'),
            #     nn.BatchNorm2d(num_features=inp_chn//2, eps=1e-5, momentum=0.1, affine=True)
            # )
        else:
            self.upconv = nn.ConvTranspose2d(inp_chn//2, inp_chn//2, 2, stride=2)
            # self.upconv = nn.Sequential(
            #     nn.ReLU(inplace=True),
            #     nn.ConvTranspose2d(inp_chn//2, inp_chn//2, kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm2d(num_features=inp_chn//2, eps=1e-5, momentum=0.1, affine=True)
            # )

        # self.conv = DoubleConv(inp_chn, out_chn)
        self.conv = nn.Sequential(
            nn.Conv2d(inp_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class FeatConv(nn.Module):
    def __init__(self, inp_chn, out_chn):
        super(FeatConv, self).__init__()
        self.featconv = nn.Conv2d(inp_chn, out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.featconv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, inp_chn, out_chn, kernel_size=4, stride=2, padding=1):
        super(OutConv, self).__init__()
        self.oconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inp_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.oconv(x)
        return x

from lib.models.networks import NetG
from lib.models.networks import UnetGenerator, define_G
from options import Options

opt = Options().parse()
device = torch.device("cuda:0" if opt.gpu_ids != -1 else "cpu")

inp = torch.rand(size=(16, 3, 32, 32), dtype=torch.float32, device=device)
netg = NetG_32().to(device)
