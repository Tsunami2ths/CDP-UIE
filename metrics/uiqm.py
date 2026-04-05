#!/usr/bin/env python3
"""
compute_uiqm_average.py

Compute the average Underwater Image Quality Measure (UIQM) for all images in a specified folder.
Usage:
    python compute_uiqm_average.py [folder_path]
    If no folder_path is provided, defaults to G:\\PCL-Net\\results\\experiment_name\\pred_cc
"""

import os
import argparse
import math
from scipy import ndimage
from PIL import Image
import numpy as np

# --- UIQM helper functions ---
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    weight = 1.0 / (K - T_a_L - T_a_R)
    s = int(T_a_L)
    e = int(K - T_a_R)
    return weight * sum(x[s:e])

def s_a(x, mu):
    return sum((pixel - mu) ** 2 for pixel in x) / len(x)

def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = (R + G) / 2 - B
    mu_RG = mu_a(RG)
    mu_YB = mu_a(YB)
    s_RG = s_a(RG, mu_RG)
    s_YB = s_a(YB, mu_YB)
    l = math.sqrt(mu_RG**2 + mu_YB**2)
    r = math.sqrt(s_RG + s_YB)
    return -0.0268 * l + 0.1586 * r

def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    return mag * (255.0 / np.max(mag)) if np.max(mag) != 0 else mag

def eme(x, window_size):
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    w = 2.0 / (k1 * k2)
    x_cropped = x[: window_size * k2, : window_size * k1]
    val = 0.0
    for i in range(k2):
        for j in range(k1):
            block = x_cropped[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ > 0 and max_ > 0:
                val += math.log(max_ / min_)
    return w * val

def _uism(x):
    R, G, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    Rs, Gs, Bs = sobel(R), sobel(G), sobel(B)
    r_eme = eme(Rs * R, 10)
    g_eme = eme(Gs * G, 10)
    b_eme = eme(Bs * B, 10)
    return 0.299 * r_eme + 0.587 * g_eme + 0.144 * b_eme

def plip_g(g, mu=1026.0):
    return mu - g

def plip_phi(g, plip_lambda=1026.0, plip_beta=1.0):
    return -plip_lambda * math.log(1 - g / plip_lambda)

def plip_phiInverse(g, plip_lambda=1026.0, plip_beta=1.0):
    return plip_lambda * (1 - math.exp(-g / plip_lambda))

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))

def _uiconm(x, window_size):
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    w = -1.0 / (k1 * k2)
    x_cropped = x[: window_size * k2, : window_size * k1, :]
    val = 0.0
    for i in range(k2):
        for j in range(k1):
            block = x_cropped[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size, :]
            max_ = np.max(block)
            min_ = np.min(block)
            if max_ + min_ > 0 and not math.isnan(max_ - min_):
                ratio = (max_ - min_) / (max_ + min_)
                val += ratio * math.log(ratio)
    return w * val

def getUIQM(x):
    x = x.astype(np.float32)
    uicm_val = _uicm(x)
    uism_val = _uism(x)
    uiconm_val = _uiconm(x, 10)
    return 0.0282 * uicm_val + 0.2953 * uism_val + 3.5753 * uiconm_val

# --- Main processing ---
def compute_uiqm_for_image(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    return getUIQM(arr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute average UIQM for images in a folder'
    )
    parser.add_argument(
        'folder',
        nargs='?',
        default=r'G:\\PCL-Net\\results\\experiment_name\\pred_cc',
        help='Directory containing image files (default: %(default)s)'
    )
    args = parser.parse_args()

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    uiqm_values = []

    for root, _, files in os.walk(args.folder):
        for fname in files:
            if fname.lower().endswith(supported_exts):
                path = os.path.join(root, fname)
                try:
                    val = compute_uiqm_for_image(path)
                    uiqm_values.append(val)
                    print(f'Processed {fname}: UIQM = {val:.4f}')
                except Exception as e:
                    print(f'Error processing {fname}: {e}')

    if uiqm_values:
        avg_uiqm = sum(uiqm_values) / len(uiqm_values)
        print(f'\nProcessed {len(uiqm_values)} images')
        print(f'Average UIQM: {avg_uiqm:.4f}')
    else:
        print('No supported images found in the specified folder.')
