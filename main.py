import argparse
import os
import sys
import math
import PIL
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch import nn

from pwc.utils import estimate
from drn.utils import predict_flow
from tools.visualize import visualize_flow_heatmap, visualize_merge_heatmap
from tools.resize import resize_img, resize_shorter_side
from tools.face_detection import detect_face


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify", required=True, help="the input of modified image")
    parser.add_argument("--origin", required=True, help="the input of original image")
    parser.add_argument("--model", required=True, help="the path to the drn model")
    parser.add_argument("--no_crop", action="store_true",
        help="do not use a face detector, instead run on the full input image")
    parser.add_argument("--output_dir", required=True, 
        help="the output directory of visualization")
    args = parser.parse_args()

    # DRN testing
    predicted_flow = predict_flow(args.modify, args.no_crop, model_path=args.model).cpu().numpy()
    predicted_flow = np.transpose(predicted_flow, (1, 2, 0))
    h, w, d = predicted_flow.shape

    if args.no_crop:
        img = Image.open(args.modify).convert('RGB')
    else:
        img, box = detect_face(args.modify)
    modified = resize_img(img, w, h)[0]
    modified_np = np.asarray(modified)
    visualize_flow_heatmap(predicted_flow, os.path.join(args.output_dir, 'drn_flow_heatmap.jpg'))
    visualize_merge_heatmap(modified_np, predicted_flow, os.path.join(args.output_dir, 'drn_merge_heatmap.jpg'))

    # PWC testing
    flow = estimate(args.modify, args.origin, args.no_crop, box, w, h).cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))
    fh, fw, fd = flow.shape

    assert(h == fh)
    assert(w == fw)
    print(f"flow: h = {fh}, w = {fw}, d = {d}, fd = {fd}")

    visualize_flow_heatmap(flow, os.path.join(args.output_dir, 'pwc_flow_heatmap.jpg'), 7.0)
    visualize_merge_heatmap(modified_np, flow, os.path.join(args.output_dir, 'pwc_merge_heatmap.jpg'), 7.0)
