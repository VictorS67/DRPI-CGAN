import pdb
import math
import torch
import torch.nn as nn

BatchNorm = nn.BatchNorm2d


class GeneratorConfig:

    num_out_channels: int = 2


class DiscriminatorConfig:

    num_in_channels: int = 2
    num_filters: int = 64
    kernels: int = 4
    stride: int = 2
    padding: int = 1


class Generator(nn.Module):

    def __init__(self, config: GeneratorConfig):
        super().__init__()

        assert(config.num_out_channels == 2)

        self.model = DRNSeg(config.num_out_channels)

    def forward(self, x):
        return self.model(x)[0]


class Discriminator(nn.Module):

    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()

        assert(config.num_in_channels == 2)
        assert(config.num_filters == 64)

        self.model = nn.Sequential(
            nn.Linear(config.num_in_channels, config.num_filters * 8),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(config.num_filters * 8, config.num_filters * 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(config.num_filters * 4, 1)
        )

    def forward(self, x):
        return nn.Sigmoid(self.model(x))


class GANModel(nn.Module):
    """A Basic GAN Model"""

    def __init__(self, gen_config: GeneratorConfig, dis_config: DiscriminatorConfig):
        self.generator = Generator(gen_config)
        self.discriminator = Discriminator(dis_config)
