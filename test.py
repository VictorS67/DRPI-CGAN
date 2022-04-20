import argparse
import os
import sys
import math
import PIL
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from typing import Optional
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable

from pwc.utils import estimate
from drn.utils import predict_flow
from drn.utils import preprocess as drn_preprocess
from tools.visualize import *
from tools.resize import resize_img, resize_shorter_side, crop_img
from tools.face_detection import detect_face
from tools.warp import warp
from prediction.model import GeneratorConfig, DiscriminatorConfig, Generator, Discriminator
from prediction.modules import ConvGANDiscriminator
from prediction.losses import BCELossFunction, LSLossFunction, MixedGenLossFunction, LossConfig
from prediction.metrics import EvaluationMetric
from dataset import GANDataset, gan_collate

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
    predicted_flow = predict_flow(args.modify, args.no_crop, model_path=args.model).detach().cpu().numpy()
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
    visualize_warp(modified_np, predicted_flow, os.path.join(args.output_dir, 'drn_wrapped.jpg'))

    # PWC testing
    flow = estimate(args.modify, args.origin, args.no_crop, box, w, h).detach()
    flow = flow.cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))
    fh, fw, fd = flow.shape

    assert(h == fh)
    assert(w == fw)
    print(f"flow: h = {fh}, w = {fw}, d = {d}, fd = {fd}")

    visualize_flow_heatmap(flow, os.path.join(args.output_dir, 'pwc_flow_heatmap.jpg'), 7.0)
    visualize_merge_heatmap(modified_np, flow, os.path.join(args.output_dir, 'pwc_merge_heatmap.jpg'), 7.0)
    visualize_warp(modified_np, flow, os.path.join(args.output_dir, 'pwc_wrapped.jpg'))

    o_img = Image.open(args.origin).convert('RGB')
    if not args.no_crop:
        o_img, _ = crop_img(o_img, box)
    o_img = resize_img(o_img, w, h)[0]
    o_img.save(os.path.join(args.output_dir, 'reshaped_original.jpg'), quality=90)
    modified.save(os.path.join(args.output_dir, 'reshaped_modified.jpg'), quality=90)
