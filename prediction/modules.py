import pdb
import math
import torch
import torch.nn as nn

from prediction.model import Discriminator, DiscriminatorConfig
# from drn.segment import DRNSub

BatchNorm = nn.BatchNorm2d


class ConvGANDiscriminator(Discriminator):

    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__(config)

        assert(config.num_in_channels == 2)
        assert(config.num_filters == 64)
        assert(config.kernels == 4)
        assert(config.stride == 2)
        assert(config.padding == 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=config.num_in_channels, out_channels=config.num_filters, kernel_size=config.kernels, stride=config.stride, padding=config.padding),
            nn.LeakyReLU(inplace=False, negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=config.num_filters, out_channels=config.num_filters * 2, kernel_size=config.kernels, stride=config.stride, padding=config.padding),
            BatchNorm(config.num_filters * 2),
            nn.LeakyReLU(inplace=False, negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=config.num_filters * 2, out_channels=config.num_filters * 4, kernel_size=config.kernels, stride=config.stride, padding=config.padding),
            BatchNorm(config.num_filters * 4),
            nn.LeakyReLU(inplace=False, negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=config.num_filters * 4, out_channels=config.num_filters * 8, kernel_size=config.kernels, stride=config.stride, padding=config.padding),
            BatchNorm(config.num_filters * 8),
            nn.LeakyReLU(inplace=False, negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=config.num_filters * 8, out_channels=1, kernel_size=config.kernels, stride=1, padding=config.padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv = self.conv4(conv)
        conv = self.conv5(conv)

        return conv
