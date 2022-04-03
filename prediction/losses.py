import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def binary_cross_entropy_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the binary cross-entropy loss.

    Args:
        targets: A [batch_size x 1] tensor, containing ground truth labels (e.g. 1 for real, 0 for fake).
        predictions: A [batch_size x 1] tensor, containing the predictions.
    
    Return:
        A scalar binary cross-entropy between targets and predictions.
    """
    bceloss = targets * torch.log(predictions) + (1.0 - targets) * torch.log(1.0 - predictions)
    return -torch.mean(bceloss)


class GeneratorLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, classify_modified: Tensor, fake: Tensor) -> Tensor:
        loss = binary_cross_entropy_loss(classify_modified, fake)

        return loss


class DiscriminatorLossFunction(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, classify_original: Tensor, classify_modified: Tensor, real: Tensor, fake: Tensor) -> Tensor:
        real_loss = binary_cross_entropy_loss(classify_original, real)
        fake_loss = binary_cross_entropy_loss(classify_modified, fake)

        return (real_loss + fake_loss) / 2
