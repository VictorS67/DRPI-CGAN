import math
import numpy as np
import PIL
import torch
import torch.nn as nn

from PIL import Image

from pwc.pwc import pwc_net
from tools.resize import resize_shorter_side, resize_img, crop_img

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load(img_path, no_crop, box, w, h):
    img = Image.open(img_path)
    if not no_crop:
        img, box = crop_img(img, box)
    img = resize_img(img, w, h)[0]
    img_numpy = np.array(img)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    return torch.FloatTensor(np.ascontiguousarray(img_numpy)).to(device)


def preprocess(img_tensor):
    _, h, w = img_tensor.shape
    preprocess_img = img_tensor.view(1, 3, h, w)
    preprocess_w, preprocess_h = int(math.floor(math.ceil(w / 64.0) * 64.0)), int(math.floor(math.ceil(h / 64.0) * 64.0))

    return h, w, preprocess_h, preprocess_w, nn.functional.interpolate(preprocess_img, size=(preprocess_h, preprocess_w), mode='bilinear', align_corners=False)


def estimate(img1, img2, no_crop, box, w, h):
    img1_tensor, img2_tensor = load(img1, no_crop, box, w, h), load(img2, no_crop, box, w, h)

    assert(img1_tensor.shape[1] == img2_tensor.shape[1])
    assert(img1_tensor.shape[2] == img2_tensor.shape[2])

    h, w, preprocess_h, preprocess_w, img1_tensor = preprocess(img1_tensor)
    _, _, _, _, img2_tensor = preprocess(img2_tensor)

    model = pwc_net('default').to(device)
    model.eval()

    predict_flow = nn.functional.interpolate(model(img1_tensor, img2_tensor), size=(h, w), mode='bilinear', align_corners=False)
    predict_flow[:, 0, :, :] *= float(w) / float(preprocess_w)
    predict_flow[:, 1, :, :] *= float(h) / float(preprocess_h)

    return predict_flow[0, :, :, :]
