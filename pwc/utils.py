import math
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

from pwc.pwc import pwc_net
from tools.resize import resize_shorter_side, resize_img, crop_img


def load(img_path, no_crop, box, w, h):
    img = Image.open(img_path)
    if not no_crop:
        img, box = crop_img(img, box)
    img = resize_img(img, w, h)[0]

    return img


def preprocess(img):
    tf = transforms.Compose([
        transforms.Lambda(lambda x: np.array(x)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)),
        transforms.Lambda(lambda x: torch.FloatTensor(np.ascontiguousarray(x)))
    ])
    # img_numpy = np.array(img)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    # img_tensor = torch.FloatTensor(np.ascontiguousarray(img_numpy)).to(device)
    img_tensor = tf(img)
    _, h, w = img_tensor.shape
    preprocess_img = img_tensor.view(1, 3, h, w)
    preprocess_w, preprocess_h = int(math.floor(math.ceil(w / 64.0) * 64.0)), int(math.floor(math.ceil(h / 64.0) * 64.0))

    return h, w, preprocess_h, preprocess_w, nn.functional.interpolate(preprocess_img, size=(preprocess_h, preprocess_w), mode='bilinear', align_corners=False)


def estimate(m_img1, o_img2, no_crop, box, w, h):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    m_img1, o_img2 = load(m_img1, no_crop, box, w, h), load(o_img2, no_crop, box, w, h)

    h, w, preprocess_h, preprocess_w, m_img1_tensor = preprocess(m_img1)
    _, _, _, _, o_img2_tensor = preprocess(o_img2)

    m_img1_tensor = m_img1_tensor.to(device)
    o_img2_tensor = o_img2_tensor.to(device)

    # print(f"img1_tensor: {img1_tensor.shape}, img2_tensor: {img2_tensor.shape}")

    model = pwc_net('default').to(device)
    model.eval()

    predict_flow = nn.functional.interpolate(model(o_img2_tensor, m_img1_tensor), size=(h, w), mode='bilinear', align_corners=False)
    predict_flow[:, 0, :, :] *= float(w) / float(preprocess_w)
    predict_flow[:, 1, :, :] *= float(h) / float(preprocess_h)

    return predict_flow[0, :, :, :]
