import PIL
import cv2

import PIL.Image as Image
import numpy as np


def visualize_warp(im, flow, output_path, alpha=1, interp=cv2.INTER_CUBIC):
    height, width, _ = flow.shape
    cart = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    pixel_map = (cart + alpha * flow).astype(np.float32)
    warped = cv2.remap(
        im,
        pixel_map[:, :, 0],
        pixel_map[:, :, 1],
        interp,
        borderMode=cv2.BORDER_REPLICATE)
    Image.fromarray(warped).save(output_path, quality=90)


def visualize_flow_heatmap(flow, output_path, max_flow_mag=7.0):
    min_flow_mag = 0.5
    flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    cv_magn = np.clip(
        255 * (flow_magn - min_flow_mag) / (max_flow_mag - min_flow_mag), 
        a_min=0, 
        a_max=255
    ).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1]

    heatmap_alpha = np.clip(flow_magn / max_flow_mag, a_min=0, a_max=1)[:, :, None]**.7
    heatmap_alpha[heatmap_alpha < .2]**.5
    pm_hm = heatmap_img * heatmap_alpha
    cv_out = np.clip(pm_hm, a_min=0, a_max=255).astype(np.uint8)
    out = Image.fromarray(cv_out)
    out.save(output_path, quality=95)


def visualize_merge_heatmap(img, flow, output_path, max_flow_mag=7.0):
    min_flow_mag = 0.5
    flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    cv_magn = np.clip(
        255 * (flow_magn - min_flow_mag) / (max_flow_mag - min_flow_mag), 
        a_min=0, 
        a_max=255
    ).astype(np.uint8)

    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1]

    h, w = cv_magn.shape
    img_alpha = np.ones((h, w), dtype=np.double)[:, :, None]

    heatmap_alpha = np.clip(flow_magn / max_flow_mag, a_min=0, a_max=1)[:, :, None]**.7
    heatmap_alpha[heatmap_alpha < .2]**.5
    pm_hm = heatmap_img * heatmap_alpha
    pm_img = img * img_alpha
    cv_out = pm_hm + pm_img * (1 - heatmap_alpha)
    cv_out = np.clip(cv_out, a_min=0, a_max=255).astype(np.uint8)
    out = Image.fromarray(cv_out)
    out.save(output_path, quality=95)


def visualize_flow_heatmap_batched(flows, output_path, max_flow_mag=7.0):
    _, h, _, d = flows.shape
    concat_flow = np.transpose(flows, (1, 2, 3, 0)).reshape(h, -1, d)

    visualize_flow_heatmap(concat_flow, output_path, max_flow_mag)


def visualize_merge_heatmap_batched(imgs, flows, output_path, max_flow_mag=7.0):
    _, ih, _, id = imgs.shape
    concat_img = np.transpose(imgs, (1, 2, 3, 0)).reshape(ih, -1, id)

    _, fh, _, fd = flows.shape
    concat_flow = np.transpose(flows, (1, 2, 3, 0)).reshape(fh, -1, fd)

    visualize_merge_heatmap(concat_img, concat_flow, output_path, max_flow_mag)


def visualize_warp_batched(imgs, flows, output_path):
    _, ih, _, id = imgs.shape
    concat_img = np.transpose(imgs, (1, 2, 3, 0)).reshape(ih, -1, id)

    _, fh, _, fd = flows.shape
    concat_flow = np.transpose(flows, (1, 2, 3, 0)).reshape(fh, -1, fd)

    visualize_warp(concat_img, concat_flow, output_path)
