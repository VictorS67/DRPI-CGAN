import torch
import numpy
from PIL import Image
from typing import Tuple


class ImageType(str):

    MODIFIED = "modified"
    ORIGINAL = "reference"


class ImageData:

    name: str
    suffix: str
    img: Image.Image
    flow: numpy.ndarray
    data: torch.Tensor
    box: Tuple
    input_sizes: Tuple
    preprocess_sizes: Tuple
