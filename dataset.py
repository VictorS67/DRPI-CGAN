import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from torch import Tensor

from datasets.dataset import Dataset, DatasetConfig


class GANDataset:

    def __init__(self, basepath: str, no_crop: bool = False) -> None:
        self.dataset = Dataset(DatasetConfig(basepath, no_crop))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str, str, Image, Image]:
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

        return (
            image_pair.modified.data, 
            image_pair.original.data, 
            image_pair.modified.name, 
            image_pair.original.name,
            image_pair.modified.img,
            image_pair.original.img,
        )

    def __len__(self) -> int:
        return len(self.dataset)


def gan_collate(
    batch: List[Tuple[Tensor, Tensor, str, str, Image, Image]]
) -> Tuple[Tensor, Tensor, List[str], List[str], List[Image], List[Image]]:
    modified_imgs, original_imgs, modified_path, original_path, modified_images, original_images = list(zip(*batch))
    return (
        torch.stack(modified_imgs), 
        torch.stack(original_imgs), 
        modified_path, 
        original_path,
        modified_images,
        original_images,
    )
