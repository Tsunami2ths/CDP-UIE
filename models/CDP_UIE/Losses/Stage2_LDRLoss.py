from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CDP_UIE.Public.loss.ssim_loss import SSIMLoss
from models.CDP_UIE.Public.loss.vgg19cr_loss import ContrastLoss
from models.CDP_UIE.Public.util.LAB2RGB_v2 import Lab2RGB

# ----------------------- DWT及辅助函数 ---------------------------
def dwt_init(x):
    """
    Haar 小波正变换。
    输入 x: [N,C,H,W]
    输出: LL, HL, LH, HH 高频分量
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4  # 低频分量
    x_HL = -x1 - x2 + x3 + x4  # 水平高频分量
    x_LH = -x1 + x2 - x3 + x4  # 垂直高频分量
    x_HH = x1 - x2 - x3 + x4  # 对角高频分量
    return x_LL, x_HL, x_LH, x_HH

def iwt(coeffs):
    """
    逆离散小波变换 (IWT)
    coeffs: tuple (LL, HL, LH, HH)
    """
    LL, HL, LH, HH = coeffs
    x1 = (LL - HL - LH + HH) / 4
    x2 = (LL - HL + LH - HH) / 4
    x3 = (LL + HL - LH - HH) / 4
    x4 = (LL + HL + LH + HH) / 4

    N, C, H, W = LL.shape
    out = torch.zeros((N, C, H * 2, W * 2), device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x1
    out[:, :, 1::2, 0::2] = x2
    out[:, :, 0::2, 1::2] = x3
    out[:, :, 1::2, 1::2] = x4
    return out

# ----------------------- LDR Loss ---------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon

    def forward(self, X, Y):
        diff = X - Y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

class HighWaveLoss(nn.Module):
    """
    高频小波一致性损失，计算 LH/HL/HH 子带 L1
    """
    def __init__(self):
        super(HighWaveLoss, self).__init__()

    def forward(self, out, ref):
        _, out_HL, out_LH, out_HH = dwt_init(out)
        _, ref_HL, ref_LH, ref_HH = dwt_init(ref)
        loss = F.l1_loss(out_HL, ref_HL) + F.l1_loss(out_LH, ref_LH) + F.l1_loss(out_HH, ref_HH)
        return loss

class LDRLoss(nn.Module):
    def __init__(self, lambda_char=1.0, lambda_ssim=0.2, lambda_hwave=0.8, lambda_local=0.1):
        super(LDRLoss, self).__init__()
        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.high_wave_loss = HighWaveLoss()
        self.local_ctr_loss = ContrastLoss(loss_weight=1.0)  # 局部对比损失

        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_hwave = lambda_hwave
        self.lambda_local = lambda_local

        self.losses = OrderedDict()
        self.lab2rgb = Lab2RGB()

    def forward(self, mid, out, ref):
        """
        mid: Stage1 输出 I_mid，用作局部对比负样本
        out: Stage2 局部重构输出 I_out
        ref: 参考图像 I_ref
        """
        # Charbonnier
        lss_char = self.char_loss(out, ref)

        # SSIM
        lss_ssim = self.ssim_loss(out, ref)

        # High-wave 高频小波一致性
        lss_hwave = self.high_wave_loss(out, ref)

        # Local contrast (局部层级对比)
        out_021 = (out + 1)/2
        ref_021 = (ref + 1)/2
        mid_021 = (mid + 1)/2

        out_rgb = self.lab2rgb.lab_to_rgb(out_021)
        ref_rgb = self.lab2rgb.lab_to_rgb(ref_021)
        mid_rgb = self.lab2rgb.lab_to_rgb(mid_021)

        lss_local = self.local_ctr_loss(out_rgb, ref_rgb, mid_rgb)

        # 保存各项损失
        self.losses['char_loss'] = lss_char
        self.losses['ssim_loss'] = lss_ssim
        self.losses['high_wave_loss'] = lss_hwave
        self.losses['local_ctr_loss'] = lss_local

        # 总损失
        total_loss = self.lambda_char * lss_char + \
                     self.lambda_ssim * lss_ssim + \
                     self.lambda_hwave * lss_hwave + \
                     self.lambda_local * lss_local

        return total_loss

    def get_losses(self):
        return self.losses
