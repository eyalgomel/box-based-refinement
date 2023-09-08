import torch
import torch.nn as nn
import torch.nn.functional as F

from bbr.models.vgg16 import VGG16


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, func=None):
        super(UpBlock, self).__init__()
        d = drop
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(d)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(d)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in):
        x = self.Upsample(x_in)
        x = self.conv1_drop(self.conv1(x))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        if self.func == "None":
            return x
        elif self.func == "tanh":
            return F.tanh(self.BN2(x))
        elif self.func == "relu":
            return F.relu(self.BN2(x))


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv1_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.Upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.BN2(x))
        return x


class MMDecoder(nn.Module):
    def __init__(self, full_features, out_channel, out_size, is_blip=False):
        super(MMDecoder, self).__init__()
        if is_blip:
            self.bottleneck = BottleneckBlock(full_features[4], 256)
            self.up0 = UpBlock(256, full_features[3], func="relu", drop=0).cuda()
            self.up1 = UpBlock(full_features[3], out_channel, func="None", drop=0).cuda()
        else:
            self.bottleneck = BottleneckBlock(full_features[4], 512)
            self.up0 = UpBlock(512, full_features[3], func="relu", drop=0).cuda()
            self.up1 = UpBlock(full_features[3], out_channel, func="None", drop=0).cuda()

        self.out_size = out_size

    def forward(self, z, z_text):
        zz = self.bottleneck(z)
        zz_norm = zz / zz.norm(dim=1).unsqueeze(dim=1)
        attn_map = (zz_norm * z_text.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdims=True)
        zz = zz * attn_map
        zz = self.up0(zz)
        zz = self.up1(zz)
        zz = F.interpolate(zz, size=self.out_size, mode="bilinear", align_corners=True)
        return torch.sigmoid(zz)


class MultiModel(nn.Module):
    def __init__(self, image_size: int):
        super(MultiModel, self).__init__()
        self.E = VGG16()
        self.D = MMDecoder(
            self.E.full_features,
            out_channel=1,
            out_size=(image_size, image_size),
            is_blip=False,
        )

    def forward(self, image, z_text):
        z_image = self.E(image)
        mask = self.D(z_image, z_text)
        return mask
