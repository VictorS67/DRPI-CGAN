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

        # assert(image_pair.modified.input_sizes[0] == image_pair.original.input_sizes[0])
        # assert(image_pair.modified.input_sizes[1] == image_pair.original.input_sizes[1])

        # input_sizes = torch.tensor([
        #     image_pair.original.input_sizes[0],
        #     image_pair.original.input_sizes[1]
        # ], dtype=torch.int32)

        # preprocess_sizes = torch.tensor([
        #     image_pair.original.preprocess_sizes[0],
        #     image_pair.original.preprocess_sizes[1]
        # ], dtype=torch.int32)

        # print(f"modified tensor: {image_pair.modified.data} - {image_pair.modified.data.shape}")
        # # print(f"original tensor: {image_pair.original.data} - {image_pair.original.data.shape}")
        # print(f"modified image: {image_pair.modified.img}")
        # print(f"modified name: {image_pair.modified.name}")
        # print(f"original name: {image_pair.original.name}")
        # # print(f"original image: {image_pair.original.img}")

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
    # print(f"modified_data: {modified_data}")
    # print(f"original_data: {original_data}")
    return (
        torch.stack(modified_tensors), 
        modified_data,
        original_data
    )
