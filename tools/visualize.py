import PIL
import cv2

import PIL.Image as Image
import numpy as np


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
    m,n = flows.shape[::2]
    concat_flow = np.rollaxis(flows,3,1).reshape(m,-1,n)

    visualize_flow_heatmap(concat_flow, output_path, max_flow_mag)


def visualize_merge_heatmap_batched(imgs, flows, output_path, max_flow_mag=7.0):
    i, j = imgs.shape[::2]
    concat_img = np.rollaxis(imgs,3,1).reshape(i, -1, j)

    m, n = flows.shape[::2]
    concat_flow = np.rollaxis(flows,3,1).reshape(m, -1, n)

    visualize_merge_heatmap(concat_img, concat_flow, output_path, max_flow_mag)
