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


class BCELossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss()

    # @torch.enable_grad()
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss(predictions, targets)

        return loss


class LSLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:

        return torch.mean((predictions - targets) ** 2)


class MixedGenLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lambda_epe = 0.1
        self.lambda_warp = 0.1
        self.lambda_gen = 0.1
        self.gen_loss = nn.BCELoss()

    def forward(
        self,
        prediction: Tensor,
        targets: Tensor,
        predicted_warp: Tensor,
        original: Tensor,
        predicted_real: Tensor,
        real: Tensor
    ) -> Tensor:

        loss_epe = torch.mean(torch.sqrt(torch.sum((prediction - targets) ** 2, dim=1)))
        loss_warp = torch.mean(torch.abs(predicted_warp - original))
        loss_gen = self.gen_loss(predicted_real, real)

        # print(f"loss_epe: {loss_epe}, loss_warp: {loss_warp}, loss_gen: {loss_gen}")

        return self.lambda_epe * loss_epe + self.lambda_warp * loss_warp + self.lambda_gen * loss_gen
