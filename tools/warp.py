import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def warp(image, flow):
    B, C, H, W = image.shape

    h_range = torch.arange(0, H).view(1, -1).repeat(1, W)
    w_range = torch.arange(0, W).view(-1, 1).repeat(H, 1)

    h_range = h_range.view(1, 1, H, W).repeat(B, 1, 1, 1)
    w_range = w_range.view(1, 1, H, W).repeat(B, 1, 1, 1)

    grid = torch.cat((w_range, h_range), 1).float().to(device)
    vgrid = Variable(grid) + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    warpped = nn.functional.grid_sample(image, vgrid.permute(0, 2, 3, 1))

    mask = Variable(torch.ones(image.shape)).to(device)
    mask = nn.functional.grid_sample(mask, vgrid.permute(0, 2, 3, 1))

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return warpped * mask
