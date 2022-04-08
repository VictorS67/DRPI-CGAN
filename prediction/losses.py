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


class GeneratorLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss()

    # @torch.enable_grad()
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss(predictions, targets)

        return loss


class DiscriminatorLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, classify_original: Tensor, classify_modified: Tensor) -> Tensor:
        real_loss = binary_cross_entropy_loss(classify_original, True)
        fake_loss = binary_cross_entropy_loss(classify_modified, False)

        return (real_loss + fake_loss) / 2
