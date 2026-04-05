import cv2
import math
import numpy as np
import kornia.color as color
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
import os


def uciqe(image):
    image = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180
    # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    # 求亮度对比值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    v = -v
    v.sort()
    v = -v
    top = np.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe


# improve the following code to support batch operation:

def torch_uciqe(image):
    # RGB转为HSV
    hsv = color.rgb_to_hsv(image)
    H, S, V = torch.chunk(hsv, 3)

    # 色度的标准差
    delta = torch.std(H) / (2 * math.pi)

    # 饱和度的平均值
    mu = torch.mean(S)

    # 求亮度对比值
    n, m = V.shape[1], V.shape[2]
    number = math.floor(n * m / 100)
    v = V.flatten()
    v, _ = v.sort()
    bottom = torch.sum(v[:number]) / number
    v = -v
    v, _ = v.sort()
    v = -v
    top = torch.sum(v[:number]) / number
    conl = top - bottom
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    return uciqe



if __name__ == '__main__':
    # 指定你要计算的文件夹路径
    folder_path = r'G:\PCL-Net\results\experiment_name\pred_cc'
    # 支持的图片后缀
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    total_score = 0.0
    img_count = 0

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(img_exts):
            img_path = os.path.join(folder_path, fname)
            try:
                # 使用 numpy+cv2 版本
                score = uciqe(img_path)
                # 如果你想用 torch 版，请把上一行改成：
                # img = Image.open(img_path).convert('RGB')
                # tensor = to_tensor(img).unsqueeze(0)  # shape [1,3,H,W]
                # score = torch_uciqe(tensor).item()

                print(f"{fname}: UCIQE = {score:.4f}")
                total_score += score
                img_count += 1
            except Exception as e:
                print(f"处理 {fname} 时出错：{e}")

    if img_count > 0:
        avg = total_score / img_count
        print(f"\n共处理 {img_count} 张图像，平均 UCIQE = {avg:.4f}")
    else:
        print("未在指定文件夹中找到有效的图像文件。")
