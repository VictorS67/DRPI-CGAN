import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


def binary_cross_entropy_loss(predictions: Tensor, classify_real: bool) -> Tensor:
    """Compute the binary cross-entropy loss.

    Args:
        predictions: A [batch_size x 1 x H x W] tensor, containing the predictions.
        classify_real: A boolean, containing whether the prediction is classifying real.

    Return:
        A scalar binary cross-entropy between targets and predictions.
    """
    if classify_real:
        targets = torch.ones_like(predictions)
    else:
        targets = torch.zeros_like(predictions)
    bceloss = targets * torch.log(predictions) + (1.0 - targets) * torch.log(1.0 - predictions)
    return -torch.mean(bceloss)


class GeneratorLossFunction(nn.Module):

    @torch.enable_grad()
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss()

    @torch.enable_grad()
    def forward(self, classify_modified: Tensor, classify_real: bool) -> Tensor:
        if classify_real:
            targets = torch.ones_like(classify_modified)
        else:
            targets = torch.zeros_like(classify_modified)
        loss = self.loss(classify_modified, targets)
        # loss = binary_cross_entropy_loss(classify_modified, classify_real)

        return loss


class DiscriminatorLossFunction(nn.Module):

    @torch.enable_grad()
    def __init__(self) -> None:
        super().__init__()

    @torch.enable_grad()
    def forward(self, classify_original: Tensor, classify_modified: Tensor) -> Tensor:
        real_loss = binary_cross_entropy_loss(classify_original, True)
        fake_loss = binary_cross_entropy_loss(classify_modified, False)

        return (real_loss + fake_loss) / 2
