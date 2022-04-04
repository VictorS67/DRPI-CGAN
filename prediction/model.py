import pdb
import math
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from typing import List, Tuple
from torch.autograd import Variable

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

    @torch.enable_grad()
    def __init__(self, config: GeneratorConfig):
        super().__init__()

        assert(config.num_out_channels == 2)

        self.model = DRNSeg(config.num_out_channels, pretrained_drn=True)

    @torch.enable_grad()
    def forward(self, x):
        # print(f"x requires grad: {x.requires_grad}")
        x.requires_grad_(True)
        return self.model(x)

    def inference(
        self,  
        predicted_flow: torch.Tensor, 
        modified_data: List[ImageData], 
        original_data: List[ImageData],
        no_crop: float
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        flows = torch.permute(predicted_flow, (0, 2, 3, 1)) # [batch_size x H x W x D]

        # print(f"modified_data: {modified_data}")
        # print(f"flow: {flows.shape}")

        n, fh, fw, _ = flows.shape
        gt_flow = []
        modified = []
        model = pwc_net('default').to(device)
        model.eval()

        for i in range(n):
            flow = flows[i]
            modified_path = modified_data[i].name
            modified_box = modified_data[i].box
            original_path = original_data[i].name

            drn_image = pwc_utils.load(modified_path, no_crop, modified_box, fw, fh)
            pwc_image = pwc_utils.load(original_path, no_crop, modified_box, fw, fh)
            h, w, ph, pw, drn_tensor = pwc_utils.preprocess(drn_image)
            _, _, _, _, pwc_tensor = pwc_utils.preprocess(pwc_image)

            # print(f"drn_tensor: {drn_tensor.shape}, pwc_tensor: {pwc_tensor.shape}")

            predict_flow = nn.functional.interpolate(model(drn_tensor, pwc_tensor), size=(h, w), mode='bilinear', align_corners=False)
            predict_flow[:, 0, :, :] *= float(w) / float(pw)
            predict_flow[:, 1, :, :] *= float(h) / float(ph)
            gt_flow.append(torch.permute(predict_flow[0, :, :, :], (1, 2, 0)))
            modified.append(np.asarray(drn_image))

        gt_flows = torch.stack(gt_flow) # [batch_size x H x W x D]
        gt_flows.requires_grad_(True)
        modified = np.array(modified)

        return flows, gt_flows, modified


class Discriminator(nn.Module):

    @torch.enable_grad()
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

    @torch.enable_grad()
    def forward(self, x):
        # print(f"x requires grad: {x.requires_grad}")
        return Variable(self.model(x), requires_grad=True)
