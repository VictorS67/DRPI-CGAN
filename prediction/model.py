import pdb
import math
import torch
import torch.nn as nn
import numpy as np

from PIL import Image

from datasets.dataset import Dataset
from tools.resize import resize_img

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
    
    @torch.no_grad()
    def inference(
        self, 
        dataloader: Dataset, 
        predicted_flow: torch.Tensor, 
        modified_name: List[str], 
        original_name: List[str], 
        modified_images: List[Image], 
        original_images: List[Image],
    ) -> Tuple[Tesnor, Tensor, np.ndarray, np.ndarray]:
        flows = np.transpose(predicted_flow.cpu().numpy(), (0, 2, 3, 1)) # [batch_size x H x W x D]
        gt_flows = dataloader.update_original(flows, modified_name, original_name) # [batch_size x H x W x D]
        n, h, w, _ = flows.shape

        modified, original = [], []
        for i in range(n):
            modified.append(np.asarray(resize_img(modified_images[i], w, h)[0]))
            original.append(np.asarray(resize_img(original_images[i], w, h)[0]))

        return torch.tensor(flows), torch.tensor(gt_flows), np.array(modified), np.array(original)


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
