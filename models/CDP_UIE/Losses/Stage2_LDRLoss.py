from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CDP_UIE.Public.loss.ssim_loss import SSIMLoss
from models.CDP_UIE.Public.loss.vgg19cr_loss import ContrastLoss


class LaplacianLoss(nn.Module):
    """
    Laplacian Loss：利用拉普拉斯算子提取图像边缘信息，
    用 L1 损失约束增强图像与参考图像在高频细节上的差异。
    """

    def __init__(self, loss_weight=1.0):
        super(LaplacianLoss, self).__init__()
        self.loss_weight = loss_weight
        # 定义 3x3 拉普拉斯核
        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]], dtype=torch.float32)
        # 调整为 (1, 1, 3, 3) 形式，适用于单通道卷积
        self.register_buffer('laplacian_kernel', laplacian_kernel.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # 将多通道图像转为灰度
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        # 确保 laplacian_kernel 与输入在同一设备上
        kernel = self.laplacian_kernel.to(pred_gray.device)
        # 计算拉普拉斯响应（使用相同 padding 保持尺寸）
        pred_lap = F.conv2d(pred_gray, kernel, padding=1)
        target_lap = F.conv2d(target_gray, kernel, padding=1)
        loss = F.l1_loss(pred_lap, target_lap)
        return self.loss_weight * loss


class LDRLoss(nn.Module):
    def __init__(self):
        super(LDRLoss, self).__init__()
        # 初始化现有损失项
        self.loss_ssim = SSIMLoss()
        self.loss_cr = ContrastLoss(loss_weight=0.5)

        # Laplacian 高频细节损失
        self.loss_laplacian = LaplacianLoss(loss_weight=1.0)

        # 损失记录字典
        self.losses = OrderedDict()

    def forward(self, raw, ldr, ref):
        """
        raw: 原始退化图像
        ldr: 第二阶段局部重构输出 (Local Detail Reconstruction)
        ref: 清晰参考图像
        """
        # 计算 SSIM 损失
        self.lss_ssim = self.loss_ssim(ldr, ref)

        # 计算对比损失
        self.lss_cr = self.loss_cr(ldr, ref, raw)

        # 计算 Laplacian 高频细节损失
        self.lss_laplacian = self.loss_laplacian(ldr, ref)

        # 记录所有损失项
        self.losses["ldr_ssim"] = self.lss_ssim
        self.losses["ldr_cr"] = self.lss_cr
        self.losses["ldr_laplacian"] = self.lss_laplacian

        # 加权总损失
        return self.lss_ssim + self.lss_cr + self.lss_laplacian

    def get_losses(self):
        return self.losses