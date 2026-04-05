import torch
import torch.nn as nn
import torch.nn.functional as F
# from wavelet import wt, iwt

import matplotlib.pyplot as plt
import numpy as np
import os

save_path = r"C:\Users\31275\Desktop\NICE_Heatmap.png"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，如果你用Mac，可改为 ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


def visualize_combined_highfreq_heatmap(lh, hl, hh, save_path="Combined_HighFreq_Heatmap.png"):
    """
    将 LH, HL, HH 三个高频子带融合，并对比其平均池化与最大池化的空间特征响应
    """
    with torch.no_grad():
        # 提取 Batch 中的第一张特征图，并取绝对值反映边缘强度
        # shape: [C, H, W]
        lh_feat = torch.abs(lh[0].detach().cpu())
        hl_feat = torch.abs(hl[0].detach().cpu())
        hh_feat = torch.abs(hh[0].detach().cpu())

        # 【核心物理融合】: 将三个方向的高频能量叠加，获得完整的边缘轮廓
        combined_feat = lh_feat + hl_feat + hh_feat

        # 1. 模拟平均池化带来的响应特征 (按通道维度 dim=0 求均值)
        avg_map = torch.mean(combined_feat, dim=0).numpy()

        # 2. 模拟最大池化带来的响应特征 (按通道维度 dim=0 求最大值)
        max_map = torch.max(combined_feat, dim=0)[0].numpy()

        # 归一化到 [0, 1] 使得热力图对比度达到最佳
        def normalize(m):
            return (m - m.min()) / (m.max() - m.min() + 1e-8)

        avg_map = normalize(avg_map)
        max_map = normalize(max_map)

        # --- 绘制并排的伪彩色图 ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：平均池化响应
        im1 = axes[0].imshow(avg_map, cmap='jet')
        axes[0].set_title('Average Pooling Spatial Response\n(Combined LH+HL+HH)', fontsize=14)
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 右图：最大池化响应
        im2 = axes[1].imshow(max_map, cmap='jet')
        axes[1].set_title('Max Pooling Spatial Response\n(Combined LH+HL+HH)', fontsize=14)
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 融合高频热力图已成功保存至: {save_path}")

def visualize_combined_highfreq_heatmap2(lh, hl, hh, save_path="Combined_HighFreq_Heatmap.png"):
    """
    使用【全局统一标尺】对比平均池化与最大池化的特征响应
    """
    with torch.no_grad():
        lh_feat = torch.abs(lh[0].detach().cpu())
        hl_feat = torch.abs(hl[0].detach().cpu())
        hh_feat = torch.abs(hh[0].detach().cpu())

        # 融合三个高频子带
        combined_feat = lh_feat + hl_feat + hh_feat

        # 1. 计算平均池化响应
        avg_map = torch.mean(combined_feat, dim=0).numpy()
        # 2. 计算最大池化响应
        max_map = torch.max(combined_feat, dim=0)[0].numpy()

        # 【核心修改：全局统一标尺】
        # 找到两者的全局最小值和最大值，确保它们在同一套颜色标准下映射
        global_min = min(avg_map.min(), max_map.min())
        global_max = max(avg_map.max(), max_map.max())

        # --- 开始绘制 ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 强制指定 vmin 和 vmax，让颜色映射绝对公平
        im1 = axes[0].imshow(avg_map, cmap='jet', vmin=global_min, vmax=global_max)
        axes[0].set_title('平均池化空间特征响应\n(高频边缘能量被过度平滑与摊薄)', fontsize=14, pad=15)
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(max_map, cmap='jet', vmin=global_min, vmax=global_max)
        axes[1].set_title('最大池化空间特征响应\n(稀疏高频边缘被精准定位与锐化)', fontsize=14, pad=15)
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 全局标尺热力图已保存: {save_path}")
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_combined_highfreq_heatmap_pure(lh, hl, hh, save_path="Combined_HighFreq_Heatmap.png"):
    """
    生成纯净版热力图：移除所有内嵌文字，并强制 Colorbar 高度与图片完美对齐
    """
    with torch.no_grad():
        lh_feat = torch.abs(lh[0].detach().cpu())
        hl_feat = torch.abs(hl[0].detach().cpu())
        hh_feat = torch.abs(hh[0].detach().cpu())

        # 融合三个高频子带
        combined_feat = lh_feat + hl_feat + hh_feat

        # 计算响应
        avg_map = torch.mean(combined_feat, dim=0).numpy()

        # 【修正点】：加上 [0] 提取 values 张量，然后再转 numpy
        max_map = torch.max(combined_feat, dim=0)[0].numpy()

        # 全局统一标尺
        global_min = min(avg_map.min(), max_map.min())
        global_max = max(avg_map.max(), max_map.max())

        # 创建画布
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 调整长宽比让图片更紧凑

        # --- 绘制左图：平均池化 ---
        im1 = axes[0].imshow(avg_map, cmap='jet', vmin=global_min, vmax=global_max)
        axes[0].axis('off')
        # 强制颜色条与图片等高
        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)

        # --- 绘制右图：最大池化 ---
        im2 = axes[1].imshow(max_map, cmap='jet', vmin=global_min, vmax=global_max)
        axes[1].axis('off')
        # 强制颜色条与图片等高
        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)

        # 减小两张图之间的空白
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print(f"✅ 纯净版等高热力图已保存: {save_path}")

# 恒等函数（用于当通道数为16时）
def x_y(x):
    return x

# ----------------------- DWT及辅助函数 ---------------------------
def dwt_init(x):
    # Haar 小波变换实现
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
    逆离散小波变换 (IWT)。
    输入 coeffs 是一个四元组 (LL, HL, LH, HH)，
    每个张量的尺寸为 [N, C, H, W]，
    输出重构后的张量尺寸为 [N, C, H*2, W*2]。
    """
    LL, HL, LH, HH = coeffs

    # 根据正变换推导逆变换公式（Haar小波）：
    # 设 a = x1, b = x2, c = x3, d = x4，则
    #   LL = a + b + c + d
    #   HL = -a - b + c + d
    #   LH = -a + b - c + d
    #   HH = a - b - c + d
    # 可解得：
    x1 = (LL - HL - LH + HH) / 4  # 对应原图的左上角像素块
    x2 = (LL - HL + LH - HH) / 4  # 左下角
    x3 = (LL + HL - LH - HH) / 4  # 右上角
    x4 = (LL + HL + LH + HH) / 4  # 右下角

    N, C, H, W = LL.shape
    out = torch.zeros((N, C, H * 2, W * 2), device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x1
    out[:, :, 1::2, 0::2] = x2
    out[:, :, 0::2, 1::2] = x3
    out[:, :, 1::2, 1::2] = x4
    return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 不需要梯度

    def forward(self, x):
        return dwt_init(x)  # 执行DWT

# 定义PONO（Positional Normalization）函数
def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)  # 计算均值
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()  # 计算标准差
    output = (x - mean) / std  # 归一化
    return output, mean, std

# 在解码器中使用的MS（Mean and Scale）函数
def MS(x, beta, gamma):
    return x * gamma + beta  # 反归一化


# ----------------------- 基础卷积模块 ---------------------------
# 定义ConvLayer模块（带反射填充的卷积层）
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # 反射填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)  # 卷积

    def forward(self, x):
        out = self.reflection_pad(x)  # 反射填充
        out = self.conv2d(out)  # 卷积
        return out

# 定义ConvBNRe模块（卷积+BN+激活）
class ConvBNRe(nn.Module):
    def __init__(self, channels):
        super(ConvBNRe, self).__init__()
        self.conv = ConvLayer(channels, channels, kernel_size=3, stride=1)  # 卷积
        self.bn = nn.BatchNorm2d(channels)  # BN
        self.ch = channels
        if self.ch == 16:
            self.relu = x_y  # 如果通道数为16，使用恒等映射
        else:
            self.relu = nn.PReLU()  # 否则使用PReLU激活

    def forward(self, x):
        residual = x  # 残差连接
        out = self.conv(x)  # 卷积
        out = self.bn(out)  # BN
        out = self.relu(out)  # 激活
        y = out + residual  # 残差连接
        return y

# 定义ConvINRe模块（卷积+InstanceNorm+激活）
class ConvINRe(nn.Module):
    def __init__(self, channels):
        super(ConvINRe, self).__init__()
        self.conv = ConvLayer(channels, channels, kernel_size=3, stride=1)  # 卷积
        self.IN = nn.InstanceNorm2d(channels)  # InstanceNorm
        self.ch = channels
        if self.ch == 16:
            self.relu = x_y  # 如果通道数为16，使用恒等映射
        else:
            self.relu = nn.PReLU()  # 否则使用PReLU激活

    def forward(self, x):
        residual = x  # 残差连接
        out = self.conv(x)  # 卷积
        out = self.IN(out)  # InstanceNorm
        out = self.relu(out)  # 激活
        y = out + residual  # 残差连接
        return y


# ----------------------- 注意力模块 ---------------------------
# 定义PALayer_l模块（低频空间注意力）
class PALayer_l(nn.Module):
    def __init__(self, channel):
        super(PALayer_l, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, max(1,channel // 4), 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 3, padding=1, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pa(x)  # 空间注意力
        y = x * out  # 应用注意力
        return y

# 定义PALayer_h模块（高频空间注意力）
class PALayer_h(nn.Module):
    def __init__(self, channel):
        super(PALayer_h, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, max(1,channel // 4), 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 3, padding=1, bias=True),
        )

    def forward(self, x):
        out = self.pa(x)  # 空间注意力
        y = x * out  # 应用注意力
        return y

# 定义CALayer_low模块（低频通道注意力）
class CALayer_low(nn.Module):
    def __init__(self, channel, k_size=3):
        super(CALayer_low, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活

    def forward(self, x):
        y = self.avg_pool(x)  # 全局平均池化
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 1D卷积
        y = self.sigmoid(y)  # Sigmoid激活
        return x * y.expand_as(x)  # 应用通道注意力

# 定义CALayer_high模块（高频通道注意力）

class CALayer_high(nn.Module):
    def __init__(self, channel, k_size=3):
        super(CALayer_high, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活

    def forward(self, x):
        x_abs = torch.abs(x)
        y = self.avg_pool(x_abs)  # 全局最大池化



        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 1D卷积
        y = self.sigmoid(y)  # Sigmoid激活
        return x * y.expand_as(x)  # 应用通道注意力



# ----------------------- DWT_transform 模块 ---------------------------
class DWT_transform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT()  # DWT模块

        # 定义多个带归一化和残差的卷积层

        self.res_h1 = ConvBNRe(in_channels)
        self.res_h2 = ConvBNRe(in_channels)
        self.res_h3 = ConvBNRe(in_channels)

        # 定义多个通道注意力层

        self.pa_high1 = PALayer_h(in_channels)
        self.ca_high1 = CALayer_high(in_channels)
        self.pa_high2 = PALayer_h(in_channels)
        self.ca_high2 = CALayer_high(in_channels)
        self.pa_high3 = PALayer_h(in_channels)
        self.ca_high3 = CALayer_high(in_channels)


    def forward(self, x):
        LL, HL, LH, HH = self.dwt(x)  # 执行DWT

        # 对高频分量进行处理
        HL = self.res_h1(HL)
        LH = self.res_h2(LH)
        HH = self.res_h3(HH)

        visualize_combined_highfreq_heatmap_pure(LH, HL, HH, save_path=save_path)

        HL_ca = self.ca_high1(HL)
        HL_pa = self.pa_high1(HL_ca)

        LH_ca = self.ca_high2(LH)
        LH_pa = self.pa_high2(LH_ca)


        HH_ca = self.ca_high3(HH)
        HH_pa = self.pa_high3(HH_ca)


        enhanced_feature = iwt((LL, HL_pa, LH_pa, HH_pa))  # IWT反变换

        return enhanced_feature


class HighFreqGlobalAttention(nn.Module):
    """高频全局注意力：结合通道注意力和大感受野空间注意力"""

    def __init__(self, in_channels=256):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力（大感受野）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),  # 7x7卷积捕捉大范围结构
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ch_att = self.channel_att(x)  # [B,64,1,1]

        # 空间注意力
        sp_att = self.spatial_att(x)  # [B,1,H,W]

        # 融合注意力
        att = ch_att * sp_att  # [B,64,H,W]
        return x * att  # 增强高频细节


class FeatureExtractDWT_High(nn.Module):
    def __init__(self, in_channels=256, out_channels=3):
        super().__init__()
        # Step 0: 降维模块：将256维特征降至128维
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 高频处理核心模块：这里调整为接受128维输入，
        # 若 DWT_transform 内部设计与通道数无关，则传入128；否则需要相应调整其内部参数
        self.dwt_transform = DWT_transform(128)

        # 全局注意力模块，输入通道设为128
        self.global_att = HighFreqGlobalAttention(in_channels=128)

        # 尾部卷积层：从128维映射到3维，采用两层卷积加非线性激活
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),  # 反射填充保持尺寸
            nn.Conv2d(128, 32, kernel_size=3, padding=0),  # 降至64维
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ReflectionPad2d(1),  # 反射填充
            nn.Conv2d(32, out_channels, kernel_size=3, padding=0),  # 映射至3通道输出
            nn.Tanh()  # 输出限制在[-1,1]
        )

    def forward(self, x):
        # x: [B,256,H,W]
        # Step 0: 降维处理：256 -> 128
        x = self.reduce_conv(x)  # [B,128,H,W]
        # Step 1: 高频分量增强：由DWT_transform处理高频信息
        enhanced = self.dwt_transform(x)  # 输出期望为 [B,128,H,W]（依据DWT_transform设计）
        # Step 2: 全局注意力增强细节
        enhanced = self.global_att(enhanced)  # [B,128,H,W]
        # Step 3: 生成最终输出：将128维映射至3维
        output = self.tail(enhanced)  # [B,3,H,W]
        return output

