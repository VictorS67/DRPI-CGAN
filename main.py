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

from pwc.utils import estimate, load
from tools.visualize import visualize_flow_heatmap, visualize_merge_heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify", required=True, help="the input of modified image")
    parser.add_argument("--origin", required=True, help="the input of original image")
    args = parser.parse_args()

    flow = estimate(args.modify, args.origin).cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))
    h, w, _ = flow.shape

    img = Image.open(args.modify).convert('RGB')
    modified = img.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)

    visualize_flow_heatmap(flow, './outputs/flow_heatmap.jpg')
    visualize_merge_heatmap(modified_np, flow, './outputs/merge_heatmap.jpg')
