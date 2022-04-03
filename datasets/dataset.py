import torch
import torchvision.transforms as transforms
import numpy as np

from torch import nn
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

from datasets.types import ImageType, ImageData
import drn.utils as drn_utils
import pwc.utils as pwc_utils
from pwc.pwc import pwc_net

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
                    img_data.name = f / image_path
                    if f.stem == ImageType.MODIFIED:
                        img_data.img, img_data.box = self.DRNLoader.load(image_path, self.no_crop)
                        img_data.data = self.DRNLoader.preprocess(img_data.img)
                        _, _, h, w = img_data.data.shape
                        img_data.input_sizes = tuple(h, w)
                        img_data.preprocess_sizes = None
                        img_data.flow = None
                        self.DRNLoader.imgs[image_path] = img_data
                    else:
                        img_data.img = None
                        img_data.data = None
                        img_data.input_sizes = None
                        img_data.preprocess_sizes = None
                        img_data.flow = None
                        self.PWCLoader.imgs[image_path] = img_data

        return self.update_images()
        # imgs = []
        # for image_path, img_data in self.DRNLoader.imgs.items():
        #     # pwc_image = self.PWCLoader.load(image_path, self.no_crop, img_data.box, img_data.input_sizes[1], img_data.input_sizes[0])
        #     # h, w, ph, pw, pwc_tensor = self.PWCLoader.preprocess(pwc_image)
        #     # self.PWCLoader.imgs[image_path].img = pwc_image
        #     # self.PWCLoader.imgs[image_path].input_sizes = tuple(h, w)
        #     # self.PWCLoader.imgs[image_path].preprocess_sizes = tuple(ph, pw)
        #     imgs.append({
        #         ImageType.MODIFIED: img_data,
        #         ImageType.ORIGINAL: self.PWCLoader.imgs[image_path]
        #     })

        # return imgs

    def __getitem__(self, index: int) -> DatasetOutput:
        image_pair = self.imgs[index]
        output = DatasetOutput()

        output.modified = image_pair[ImageType.MODIFIED]
        output.original = image_pair[ImageType.ORIGINAL]
        return output

    def __len__(self) -> int:
        return len(self.imgs)
    
    def update_original(self, flows: np.ndarray, modified_name: List[str], original_name: List[str]) -> np.ndarray:
        n, mh, mw, _ = flows.shape
        gt_flow = []
        model = pwc_net('default').to(device)
        model.eval()
        for i in range(n):
            flow = flows[i]
            modified_path = modified_name[i]
            original_path = original_name[i]

            image_data = self.DRNLoader.imgs[modified_path]
            drn_image = self.PWCLoader.load(modified_path, self.no_crop, image_data.box, mw, mh)
            pwc_image = self.PWCLoader.load(original_path, self.no_crop, image_data.box, mw, mh)
            h, w, ph, pw, drn_tensor = self.PWCLoader.preprocess(drn_image)
            _, _, _, _, pwc_image = self.PWCLoader.preprocess(pwc_image)

            predict_flow = nn.functional.interpolate(model(pwc_tensor, drn_tensor), size=(h, w), mode='bilinear', align_corners=False)
            predict_flow[:, 0, :, :] *= float(w) / float(pw)
            predict_flow[:, 1, :, :] *= float(h) / float(ph)

            self.DRNLoader.imgs[modified_path].preprocess_sizes = tuple(mh, mw)
            self.DRNLoader.imgs[modified_path].flow = flow
            self.PWCLoader.imgs[original_path].img = pwc_image
            self.PWCLoader.imgs[original_path].input_sizes = tuple(h, w)
            self.PWCLoader.imgs[original_path].preprocess_sizes = tuple(ph, pw)
            self.PWCLoader.imgs[original_path].flow = np.transpose(predict_flow[0, :, :, :].cpu().numpy(), (1, 2, 0))
            gt_flow.append(self.PWCLoader.imgs[original_path].flow)
        
        return np.array(gt_flow)


    def update_images(self) -> List[Dict[ImageType, ImageData]]:
        imgs = {}
        for image_path, img_data in self.DRNLoader.imgs.items():
            image_name = img_data.name.split('/')[-1]
            if image_name not in imgs:
                imgs[image_name] = {}
            imgs[image_name][ImageType.MODIFIED] = img_data
        
        for image_path, img_data in self.PWCLoader.imgs.items():
            image_name = img_data.name.split('/')[-1]
            if image_name not in imgs:
                imgs[image_name] = {}
            imgs[image_name][ImageType.ORIGINAL] = img_data

        return list(imgs.values())
