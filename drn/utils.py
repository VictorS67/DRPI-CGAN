import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from drn.segment import DRNSeg
from tools.resize import resize_shorter_side, resize_img
from tools.face_detection import detect_face


def load(img_path, no_crop):
    if no_crop:
        img = Image.open(img_path).convert('RGB')
        box = img.getbbox()
    else:
        img, box = detect_face(img_path)
        img = resize_shorter_side(img, 400)[0]
    return img, box


def preprocess(img):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = tf(img)

    return img_tensor.unsqueeze(0)


def predict_flow(img, no_crop, model_path=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img, box = load(img, no_crop)
    img_tensor = preprocess(img).to(device)

    model = DRNSeg(2)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        flow = model(img_tensor)[0]

    return flow
