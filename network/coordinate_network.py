# https://github.com/anibali/dsntnn
# DSNTNN Coordinate Regression Network, extended fully convolutional network

import dsntnn
import torch.nn as nn

from network.mobilenet import mobilenetv2_ed
from network.resnet import resnet18
from network.shufflenet import shufflenetv2_ed


class CoordinatesRegressionNetwork(nn.Module):

    def __init__(self, out_size, network_model):
        super(CoordinatesRegressionNetwork, self).__init__()

        if network_model == "resnet":
            self.backend = resnet18(pretrained=False)
            self.in_size = 32
        elif network_model == "mobilenetv2":
            self.backend = mobilenetv2_ed(width_mult=1.0)
            self.in_size = 32
        elif network_model == "shufflenetv2":
            self.backend = shufflenetv2_ed(width_mult=1.0)
            self.in_size = 32
        else:
            raise ValueError("Unknown network model")

        self.heatmap_conv = nn.Conv2d(self.in_size, out_size, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        out = self.backend(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.heatmap_conv(out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps
