import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


def binary_cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute the binary cross-entropy loss.

    Args:
        predictions: A [batch_size x 1 x H x W] tensor, containing the predictions.
        classify_real: A boolean, containing whether the prediction is classifying real.

    Return:
        A scalar binary cross-entropy between targets and predictions.
    """
    bceloss = targets * torch.log(predictions) + (1.0 - targets) * torch.log(1.0 - predictions)

    return -torch.mean(bceloss)


class LossConfig:

    epe: float = 0.1
    warp: float = 0.01
    gan: float = 0.1


class LossData:

    loss_epe: float
    loss_warp: float
    loss_gan: float


class BCELossFunction(nn.Module):

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.loss = nn.BCELoss()
        self.loss_data = LossData()

    # @torch.enable_grad()
    def forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, LossData]:
        loss = self.loss(predictions, targets)
        self.loss_data.loss_gan = loss.float().item()

        return loss, self.loss_data


class LSLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_data = LossData()

    def forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, LossData]:
        loss = torch.mean((predictions - targets) ** 2)
        self.loss_data.loss_gan = loss.float().item()

        return loss, self.loss_data


class MixedGenLossFunction(nn.Module):

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.lambdas = config
        self.gen_loss = nn.BCELoss()
        self.loss_data = LossData()

    def forward(
        self,
        prediction: Tensor,
        targets: Tensor,
        predicted_warp: Tensor,
        original: Tensor,
        predicted_real: Tensor,
        real: Tensor
    ) -> Tuple[Tensor, LossData]:

        loss_epe = torch.mean(torch.sqrt(torch.sum((prediction - targets) ** 2, dim=1)))
        loss_warp = torch.mean(torch.abs(predicted_warp - original))
        loss_gan = self.gen_loss(predicted_real, real)

        # print(f"loss_epe: {loss_epe}, loss_warp: {loss_warp}, loss_gan: {loss_loss_ganen}")
        total_loss = self.lambdas.epe * loss_epe + self.lambdas.warp * loss_warp + self.lambdas.gan * loss_gan
        self.loss_data.loss_epe = loss_epe.float().item()
        self.loss_data.loss_warp = loss_warp.float().item()
        self.loss_data.loss_gan = loss_gan.float().item()

        return total_loss, self.loss_data
