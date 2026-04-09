from collections import OrderedDict
from datetime import datetime
import torch
import torch.nn as nn
from models.CDP_UIE.Public.loss.vgg19cr_loss import ContrastLoss
from models.CDP_UIE.Public.util.LAB2RGB_v2 import Lab2RGB

# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon

    def forward(self, X, Y):
        diff = X - Y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

# 小波低频一致性损失
class LLWaveLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LLWaveLoss, self).__init__()
        self.eps = epsilon

    def forward(self, new, old):
        """
        new: 小波低频处理后映射回三通道的结果 [B,C,H,W]
        old: Stage1 输出合并后的三通道 Lab 图片 [B,C,H,W]
        """
        # 假设 new 和 old 已经是同尺寸的三通道图片
        diff = new - old
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

class GRLoss(nn.Module):
    def __init__(self, lambda_char=1.0, lambda_ll=0.5, lambda_global=0.1):
        super(GRLoss, self).__init__()
        # Charbonnier 重构损失
        self.char_loss = CharbonnierLoss()
        # 小波低频一致性损失
        self.ll_wave_loss = LLWaveLoss()
        # 全局对比损失
        self.global_ctr_loss = ContrastLoss(loss_weight=1.0)  # λ3 在总损失中使用

        self.lambda_char = lambda_char
        self.lambda_ll = lambda_ll
        self.lambda_global = lambda_global

        self.losses = OrderedDict()
        self.lab2rgb = Lab2RGB()

    def forward(self, raw, old, new, ref):
        # Charbonnier 重构损失
        lss_char = self.char_loss(old, ref)

        # 小波低频一致性损失
        lss_ll = self.ll_wave_loss(new, old)

        # 全局对比损失（在 RGB 空间）
        raw_021 = (raw + 1) / 2
        new_021 = (new + 1) / 2
        ref_021 = (ref + 1) / 2

        raw_rgb = self.lab2rgb.lab_to_rgb(raw_021)
        new_rgb = self.lab2rgb.lab_to_rgb(new_021)
        ref_rgb = self.lab2rgb.lab_to_rgb(ref_021)

        lss_global = self.global_ctr_loss(new_rgb, ref_rgb, raw_rgb)

        # 存储各项损失
        self.losses['char_loss'] = lss_char
        self.losses['ll_wave_loss'] = lss_ll
        self.losses['global_ctr_loss'] = lss_global

        # 总损失
        total_loss = self.lambda_char * lss_char + \
                     self.lambda_ll * lss_ll + \
                     self.lambda_global * lss_global

        return total_loss

    def get_losses(self):
        return self.losses
