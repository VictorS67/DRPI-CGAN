import math
import numpy as np
from typing import List, Tuple
from tools.warp import warp

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


def epe(predict_flow: Tensor, gt_flow: Tensor) -> float:
    """ Evaluate End-To-End Point Error (EPE).

    args:
        - predict_flow: a [2 x H x W] tensor.
        - gt_flow: a [2 x H x W] tensor.

    return:
        - epe: a float.
    """
    epe = torch.mean(torch.sqrt(torch.sum((predict_flow - gt_flow) ** 2, dim=0))).item()

    return epe


def psnr(img1: Tensor, img2: Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    # MSE is zero means no noise is present in the signal.
    # Therefore PSNR have no importance.
    if (mse == 0):
        return 100.0
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr.item()


def delta_psnr(predict_flow: Tensor, modified_img: Tensor, original_img: Tensor) -> Tuple[float, float]:
    """ Evaluate Peak Signal-To-Noise Ratio (PSNR) for the modified image and warpped image.

    args:
        - predict_flow: a [2 x H x W] tensor.
        - modified_img: a [3 x H x W] tensor.
        - original_img: a [3 x H x W] tensor.

    return:
        - psnr_before: a float. This is the psnr for the modified image.
        - psnr_after: a float. This is the psnr for the warpped image.
    """
    warpped_img = warp(modified_img[None,:], predict_flow[None,:])[0]
    psnr_before = psnr(original_img, modified_img)
    psnr_after = psnr(original_img, warpped_img)

    return psnr_before, psnr_after


def iou(predict_flow: Tensor, gt_flow: Tensor) -> float:
    """ Evaluate the Intersection-Over-Union (IOU) of two flows.

    args:
        - predict_flow: a [2 x H x W] tensor.
        - gt_flow: a [2 x H x W] tensor.

    return:
        - iou: a float.
    """
    min_flow_mag = 0.5

    predict_flow_magn = torch.sqrt(predict_flow[0, :, :]**2 + predict_flow[1, :, :]**2) # [H, W]
    predict_flow_magn = (predict_flow_magn > min_flow_mag).float()

    gt_flow_magn = torch.sqrt(gt_flow[0, :, :]**2 + gt_flow[1, :, :]**2) # [H, W]
    gt_flow_magn = (gt_flow_magn > min_flow_mag).float()

    intersection = torch.count_nonzero((predict_flow_magn + gt_flow_magn) == 2.0).item()
    union = torch.count_nonzero((predict_flow_magn + gt_flow_magn) > 0.0).item()

    return intersection / union + 0.0


class EvaluationData:

    epes: List[float] = []
    before_psnrs: List[float] = []
    after_psnrs: List[float] = []
    ious: List[float] = []


class EvaluationMetric(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.evaldata = EvaluationData()

    def forward(self, predict_flow: Tensor, gt_flow: Tensor, modified_img: Tensor, original_img: Tensor) -> None:

        epe_data = epe(predict_flow, gt_flow)
        before_psnr_data, after_psnr_data = delta_psnr(predict_flow, modified_img, original_img) 
        iou_data = iou(predict_flow, gt_flow)

        self.evaldata.epes.append(epe_data)
        self.evaldata.before_psnrs.append(before_psnr_data)
        self.evaldata.after_psnrs.append(after_psnr_data)
        self.evaldata.ious.append(iou_data)

    def inference(self) -> Tuple[float, float, float]:
        avg_epe = np.mean(self.evaldata.epes)
        delta_psnr = np.mean(self.evaldata.after_psnrs) - np.mean(self.evaldata.before_psnrs)
        avg_iou = np.mean(self.evaldata.ious)

        return avg_epe, delta_psnr, avg_iou
