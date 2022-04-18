import math
from typing import List, Tuple
from tools.warp import warp

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

def epe(ground_truth_optical_flow: Tensor, predicted_optical_flow: Tensor)->float:
    return torch.mean(torch.sqrt(torch.sum((predicted_optical_flow - ground_truth_optical_flow) ** 2, dim=1))).item()

def psnr_helper(img1, img2)->float:
    return -10 * torch.log10 * torch.mean(img1 - img2 + 1e-6).item()

def delta_psnr(modified_image:Tensor, original_image:Tensor, predicted_optical_flow: Tensor)->float:
    unwarpped_image = warp(modified_image, predicted_optical_flow)
    psnr_before = psnr_helper(original_image/255, modified_image/255)
    psnr_after = psnr_helper(original_image/255, unwarpped_image/255)
    return psnr_before, psnr_after


def iou(ground_truth_optical_flow:Tensor, predicted_optical_flow:Tensor)->float:
    ground_truth_magn= torch.norm(ground_truth_optical_flow, p=2, dim=1, keepdim=True)
    predicted_magn=torch.norm(predicted_optical_flow, p=2, dim=1, keepdim=True)
    h = predicted_magn.size(dim=0)
    w = predicted_magn.size(dim=1)
    predicted_magn = predicted_magn.view(1,h*w)
    ground_truth_magn = ground_truth_magn.view(1,h*w)
    sum_magn = predicted_magn + ground_truth_magn
    return sum_magn[sum_magn == 2].size(dim=0)/sum_magn[sum_magn > 0].size(dim=0)


if __name__ == "__main__":
    pass