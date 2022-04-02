import torch
from PIL import Image
from typing import Tuple


class ImageType(str):

    MODIFIED = "modified"
    ORIGINAL = "reference"


class ImageData:

    name: str
    img: Image
    data: torch.Tensor
    box: Tuple
    input_sizes: Tuple
    preprocess_sizes: Tuple
