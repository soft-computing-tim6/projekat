# https://arxiv.org/pdf/1702.08502.pdf

import torch.nn as nn


class DUC(nn.Module):
    """
    Dense Upsampling Convolutional Module
    Initialize: in_planes, planes, upscale_factor
    Output: (planes // upscale_factor^2) * ht * wd
    """

    def __init__(self, in_dim, upsample_dim, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(upsample_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
