from collections import OrderedDict
from datetime import datetime

import torch.nn as nn

from models.CDP_UIE.Public.loss.vgg19cr_loss import ContrastLoss
from models.CDP_UIE.Public.util.LAB2RGB_v2 import Lab2RGB

class GRLoss(nn.Module):
    def __init__(self):
        super(GRLoss, self).__init__()
        # 定义 MSE 损失函数（均方误差损失）
        self.mseloss = nn.MSELoss()
        # 定义对比损失函数（ContrastLoss），权重为 0.005
        self.crloss = ContrastLoss(loss_weight=0.005)

        # 使用 OrderedDict 存储各项损失值
        self.losses = OrderedDict()
        # 定义 Lab 到 RGB 的转换工具
        self.lab2rgb = Lab2RGB()
        pass

    def forward(self, raw, old, new, ref):  # 假设 raw, enc, ref 是 Lab 格式，取值范围为 [-1, 1]
        start = datetime.now()



        # 提取 L 通道和 ab 通道
        old_l = old[:, 0:1, :, :]  # L 通道是 Lab 的第一个通道
        ref_l = ref[:, 0:1, :, :]
        old_ab = old[:, 1:3, :, :]  # ab 通道是 Lab 的后两个通道
        ref_ab = ref[:, 1:3, :, :]

        # 计算 L 通道的 MSE 损失
        self.lss_mse_l = self.mseloss(old_l, ref_l)
        t_mse_l = datetime.now()

        # 计算 ab 通道的 MSE 损失
        self.lss_mse_ab = self.mseloss(old_ab, ref_ab)
        t_mse_ab = datetime.now()

        # 计算对比损失（Contrast Loss）
        # 将输入从 [-1, 1] 范围转换到 [0, 1] 范围
        raw_021 = (raw + 1) / 2
        new_021 = (new + 1) / 2
        ref_021 = (ref + 1) / 2

        # 将 Lab 格式的图像转换为 RGB 格式
        raw_rgb_021 = self.lab2rgb.lab_to_rgb(raw_021)
        new_rgb_021 = self.lab2rgb.lab_to_rgb(new_021)
        ref_rgb_021 = self.lab2rgb.lab_to_rgb(ref_021)

        # 计算 new 和 ref 在 RGB 空间上的对比损失
        self.lss_cr = self.crloss(new_rgb_021, ref_rgb_021, raw_rgb_021)

        # 将各项损失值存储到字典中
        self.losses['cc_mse_l'] = self.lss_mse_l  # 存储 L 通道的 MSE 损失
        self.losses['cc_mse_ab'] = self.lss_mse_ab  # 存储 ab 通道的 MSE 损失
        self.losses['cc_cr'] = self.lss_cr    # 存储对比损失

        # 返回总损失（L 通道 MSE 损失 + ab 通道 MSE 损失 + 对比损失）
        return self.lss_mse_l + self.lss_mse_ab + self.lss_cr
        pass

    def get_losses(self):
        # 返回存储的损失值字典
        return self.losses