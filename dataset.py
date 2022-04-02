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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        image_pair = self.dataset[index]

        assert(image_pair.modified.input_sizes[0] == image_pair.original.input_sizes[0])
        assert(image_pair.modified.input_sizes[1] == image_pair.original.input_sizes[1])

        input_sizes = torch.tensor([
            image_pair.original.input_sizes[0],
            image_pair.original.input_sizes[1]
        ], dtype=torch.int32)

        preprocess_sizes = torch.tensor([
            image_pair.original.preprocess_sizes[0],
            image_pair.original.preprocess_sizes[1]
        ], dtype=torch.int32)

        return image_pair.modified.data, image_pair.original.data, input_sizes, preprocess_sizes

    def __len__(self) -> int:
        return len(self.dataset)


def gan_collate(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    modified_imgs, original_imgs, input_sizes, preprocess_sizes = list(zip(*batch))
    return torch.stack(modified_imgs), torch.stack(original_imgs), torch.stack(input_sizes), torch.stack(preprocess_sizes)
