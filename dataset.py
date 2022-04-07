import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from torch import Tensor

from datasets.dataset import Dataset, DatasetConfig
from datasets.types import ImageData


class GANDataset:

    def __init__(self, basepath: str, no_crop: bool = False) -> None:
        dataconfig = DatasetConfig()
        dataconfig.basepath = basepath
        dataconfig.no_crop = no_crop
        self.dataset = Dataset(dataconfig)

    def __getitem__(self, index: int) -> Tuple[Tensor, ImageData, ImageData]:
        image_pair = self.dataset[index]

        return (
            image_pair.modified.data, 
            image_pair.modified,
            image_pair.original
        )

    def __len__(self) -> int:
        return len(self.dataset)


def gan_collate(
    batch: List[Tuple[Tensor, ImageData, ImageData]]
) -> Tuple[Tensor, List[ImageData], List[ImageData]]:
    modified_tensors, modified_data, original_data = list(zip(*batch))

    return (
        torch.stack(modified_tensors), 
        modified_data,
        original_data
    )
