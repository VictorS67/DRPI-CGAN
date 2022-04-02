import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

from datasets.types import ImageType, ImageData
import drn.utils as drn_utils
import pwc.utils as pwc_utils


class DatasetConfig:

    basepath: str
    no_crop: bool = False


class DatasetOutput:

    modified: ImageData
    original: ImageData


class Dataset(torch.utils.data.Dataset):

    class DRNLoader:

        def __init__(self) -> None:
            self.imgs = {}

        def load(self, img_path, no_crop, box=None) -> Tuple[Image, Tuple]:
            return drn_utils.load(img_path, no_crop)

        def preprocess(self, img) -> torch.Tensor:
            return drn_utils.preprocess(img)

    class PWCLoader:

        def __init__(self) -> None:
            self.imgs = {}

        def load(self, img_path, no_crop, box, w, h) -> Image:
            return pwc_utils.load(img_path, no_crop, box, w, h)

        def preprocess(self, img) -> Tuple[int, int, int, int, torch.Tensor]:
            return pwc_utils.preprocess(img)

    def __init__(self, config: DatasetConfig) -> None:
        self.basepath = Path(config.basepath)
        self.no_crop = config.no_crop
        self.DRNLoader = DRNLoader()
        self.PWCLoader = PWCLoader()
        self.imgs = self._load_images()

    def _load_images(self) -> List[Dict[ImageType, ImageData]]:
        for f in self.basepath.iterdir():
            if f.is_dir() and (f.stem == ImageType.MODIFIED or f.stem == ImageType.ORIGINAL):
                images = sorted(f.iterdir())
                for image_path in images:
                    img_data = ImageData()
                    img_data.name = image_path.split(".")[0]
                    if f.stem == ImageType.MODIFIED:
                        img_data.img, img_data.box = self.DRNLoader.load(image_path, self.no_crop)
                        img_data.data = self.DRNLoader.preprocess(img_data.img)
                        _, _, h, w = img_data.data.shape
                        img_data.input_sizes = tuple(h, w)
                        img_data.preprocess_sizes = tuple(h, w)
                        self.DRNLoader.imgs[image_path] = img_data
                    else:
                        img_data.img = None
                        img_data.data = None
                        img_data.input_sizes = None
                        img_data.preprocess_sizes = None
                        self.PWCLoader.imgs[image_path] = img_data

        imgs = []
        for image_path, img_data in self.DRNLoader.imgs.items():
            pwc_image = self.PWCLoader.load(image_path, self.no_crop, img_data.box, img_data.input_sizes[1], img_data.input_sizes[0])
            h, w, ph, pw, pwc_tensor = self.PWCLoader.preprocess(pwc_image)
            self.PWCLoader.imgs[image_path].img = pwc_image
            self.PWCLoader.imgs[image_path].input_sizes = tuple(h, w)
            self.PWCLoader.imgs[image_path].preprocess_sizes = tuple(ph, pw)
            imgs.append({
                ImageType.MODIFIED: img_data,
                ImageType.ORIGINAL: self.PWCLoader.imgs[image_path]
            })

        return imgs

    def __getitem__(self, index: int) -> DatasetOutput:
        image_pair = self.imgs[index]
        output = DatasetOutput()

        output.modified = image_pair[ImageType.MODIFIED]
        output.original = image_pair[ImageType.ORIGINAL]
        return output

    def __len__(self) -> int:
        return len(self.imgs)
