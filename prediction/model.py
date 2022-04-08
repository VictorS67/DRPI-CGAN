import pdb
import math
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from typing import List, Tuple

from dataset import GANDataset
from datasets.types import ImageData
from tools.resize import resize_img
from drn.segment import DRNSeg
import pwc.utils as pwc_utils
from pwc.pwc import pwc_net

BatchNorm = nn.BatchNorm2d
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        super(Generator, self).__init__()

        assert(config.num_out_channels == 2)

        self.model = DRNSeg(config.num_out_channels, pretrained_drn=True)

    def forward(self, x):
        out = self.model(x)

        return out

    @torch.no_grad()
    def inference(
        self,  
        predicted_flow: torch.Tensor, 
        modified_data: List[ImageData], 
        original_data: List[ImageData],
        no_crop: float
    ) -> Tuple[torch.Tensor, np.ndarray]:

        n, _, fh, fw = predicted_flow.shape
        gt_flow = []
        modified = []
        model = pwc_net('default').to(device)
        model.eval()

        for i in range(n):
            modified_path = modified_data[i].name
            modified_box = modified_data[i].box
            original_path = original_data[i].name

            drn_image = pwc_utils.load(modified_path, no_crop, modified_box, fw, fh)
            pwc_image = pwc_utils.load(original_path, no_crop, modified_box, fw, fh)
            h, w, ph, pw, drn_tensor = pwc_utils.preprocess(drn_image)
            _, _, _, _, pwc_tensor = pwc_utils.preprocess(pwc_image)

            drn_tensor = drn_tensor.to(device)
            pwc_tensor = pwc_tensor.to(device)

            predict_flow = nn.functional.interpolate(model(pwc_tensor, drn_tensor), size=(h, w), mode='bilinear', align_corners=False)
            predict_flow[:, 0, :, :] *= float(w) / float(pw)
            predict_flow[:, 1, :, :] *= float(h) / float(ph)
            gt_flow.append(predict_flow[0, :, :, :])
            modified.append(np.asarray(drn_image))

        gt_flows = torch.stack(gt_flow)  # [batch_size x D x H x W]
        modified = np.array(modified)

        return gt_flows, modified


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
            nn.Linear(config.num_filters * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)
