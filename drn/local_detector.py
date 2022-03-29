import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import dlib
from dlib import cnn_face_detection_model_v1 as face_detect_model

from networks.drn_seg import DRNSeg
# from utils.tools import *
# from utils.visualize import *
from tools.visualize import visualize_flow_heatmap, visualize_merge_heatmap
cnn_face_detector = None
def resize_shorter_side(img, min_length):
    """
    Resize the shorter side of img to min_length while
    preserving the aspect ratio.
    """
    ow, oh = img.size
    mult = 8
    if ow < oh:
        if ow == min_length and oh % mult == 0:
            return img, (ow, oh)
        w = min_length
        h = int(min_length * oh / ow)
    else:
        if oh == min_length and ow % mult == 0:
            return img, (ow, oh)
        h = min_length
        w = int(min_length * ow / oh)
    return img.resize((w, h), Image.BICUBIC), (w, h)

def face_detection(
        img_path,
        verbose=False,
        model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    """
    Detects faces using dlib cnn face detection, and extend the bounding box
    to include the entire face.
    """
    def shrink(img, max_length=2048):
        ow, oh = img.size
        if max_length >= max(ow, oh):
            return img, 1.0

        if ow > oh:
            mult = max_length / ow
        else:
            mult = max_length / oh
        w = int(ow * mult)
        h = int(oh * mult)
        return img.resize((w, h), Image.BILINEAR), mult

    global cnn_face_detector
    if cnn_face_detector is None:
        cnn_face_detector = face_detect_model(model_file)

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img_shrinked, mult = shrink(img)

    im = np.asarray(img_shrinked)
    if len(im.shape) != 3 or im.shape[2] != 3:
        return []

    crop_ims = []
    dets = cnn_face_detector(im, 0)
    for k, d in enumerate(dets):
        top = d.rect.top() / mult
        bottom = d.rect.bottom() / mult
        left = d.rect.left() / mult
        right = d.rect.right() / mult

        wid = right - left
        left = max(0, left - wid // 2.5)
        top = max(0, top - wid // 1.5)
        right = min(w - 1, right + wid // 2.5)
        bottom = min(h - 1, bottom + wid // 2.5)

        if d.confidence > 1:
            if verbose:
                print("%d-th face detected: (%d, %d, %d, %d)" %
                      (k, left, top, right, bottom))
            crop_im = img.crop((left, top, right, bottom))
            crop_ims.append((crop_im, (left, top, right, bottom)))

    return crop_ims

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--dest_folder", required=True, help="folder to store the results")
    parser.add_argument(
        "--model_path", required=True, help="path to the drn model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()

    img_path = args.input_path
    dest_folder = args.dest_folder
    model_path = args.model_path
    gpu_id = args.gpu_id

    # Loading the model
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    # device = 'cpu'

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    # Data preprocessing
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    im_w, im_h = Image.open(img_path).size
    if args.no_crop:
        face = Image.open(img_path).convert('RGB')
    else:
        faces = face_detection(img_path, verbose=False)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)

    # Warping field prediction
    with torch.no_grad():
        flow = model(face_tens.unsqueeze(0))[0].cpu().numpy()
        flow = np.transpose(flow, (1, 2, 0))
        h, w, _ = flow.shape

    # Undoing the warps
    modified = face.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)
    # reverse_np = warp(modified_np, flow)
    # reverse = Image.fromarray(reverse_np)

    # Saving the results
    # modified.save(
    #     os.path.join(dest_folder, 'cropped_input.jpg'),
    #     quality=90)
    # reverse.save(
    #     os.path.join(dest_folder, 'warped.jpg'),
    #     quality=90)
    # flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    # save_heatmap_cv(
    #     modified_np, flow_magn,
    #     os.path.join(dest_folder, 'heatmap.jpg'))
    visualize_flow_heatmap(flow, './out/flow_heatmap.jpg')
    visualize_merge_heatmap(modified_np, flow, './out/merge_heatmap.jpg')
