import torch.nn as nn
import torch

from models.CDP_UIE.Public.net.FAB import FABlock, default_conv  # 导入FABlock模块和默认卷积
from models.CDP_UIE.Stage1_GR.L_Branch import L_Branch
from models.CDP_UIE.Stage1_GR.FeatureExtractDWT import FeatureExtractDWT
from models.CDP_UIE.Stage1_GR.CCFM import CCFM

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # 反射填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)  # 卷积

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
class TailConv(nn.Module):
    def __init__(self, in_channels=64, mid_channels=24, out_channels=3):
        super().__init__()
        self.tail = nn.Sequential(
            ConvLayer(in_channels, mid_channels, 3, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ConvLayer(mid_channels, out_channels, 1, 1),
            nn.Tanh()  # 输出限制在 [-1, 1] 范围
        )

    def forward(self, x):
        return self.tail(x)

class GatedFilter(nn.Module):
    """
    对齐后特征门控融合：
      - 先将 ab 和 rgb 特征拼接
      - 通过 1×1 卷积 + BN + ReLU + 1×1 卷积 + Sigmoid 学习一张 gate 图 (B×C×H×W)
      - 最终按 gate 对 rgb 和 ab 做加权融合
    """
    def __init__(self, channels, reduction=4):
        super(GatedFilter, self).__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            # 降维
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # 升维回原始通道数
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            # 硬 Sigmoid，加速推理
            nn.Hardsigmoid()
        )

    def forward(self, F_a, F_b):
        # F_a, F_b: [B, C, H, W]
        F_cat = torch.cat([F_a, F_b], dim=1)      # [B, 2C, H, W]
        g     = self.gate(F_cat)                  # [B,   C, H, W], ∈[0,1]
        F_a_f = g * F_a
        F_b_f = (1 - g) * F_b
        return F_a_f, F_b_f

# 定义 Inception 模块
class Inc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Inc, self).__init__()
        # 分支1：1x1卷积 + 3x3卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1), dilation=1,
                      padding=(3 - 1) // 2),
            nn.LeakyReLU(),
        )
        # 分支2：1x1卷积 + 5x5卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1), dilation=1,
                      padding=(5 - 1) // 2),
            nn.LeakyReLU(),
        )
        # 分支3：最大池化 + 1x1卷积
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )
        # 分支4：1x1卷积
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        # 分别通过四个分支
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        # 将四个分支的输出在通道维度上拼接
        return torch.cat([o1, o2, o3, o4], dim=1)


# 定义展平模块
class Flatten(nn.Module):
    def forward(self, inp):
        # 将输入张量展平为二维张量 (batch_size, -1)
        return inp.view(inp.size(0), -1)


# 定义通道注意力模块 (Channel Attention Module)
class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            # 展平
            Flatten(),
            # 全连接层，降维
            nn.Linear(in_channels, in_channels // reduction_ratio),
            # 激活函数
            nn.Softsign(),
            # 全连接层，恢复维度
            nn.Linear(in_channels // reduction_ratio, in_channels),
            # 激活函数
            nn.Softsign()
        )

    def forward(self, input):
        # 计算通道注意力权重，并与输入相乘
        return input * self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)


class GRNet(nn.Module):
    def __init__(self,
                 input_chn=2,  # 输入通道数，默认为2（ab通道）
                 feature_chn=64,  # 特征通道数，默认为64
                 output_chn=2):  # 输出通道数，默认为2（ab通道）
        super(GRNet, self).__init__()

        # 初始化参数
        self.input_chn = input_chn
        self.output_chn = output_chn

        # 初始化网络层
        self._init_layers()

    def _init_layers(self):
        chn_tmp = 64  # 中间特征通道数

        # ---------------------------Lab 分支：处理 ab 通道 ---------------------------
        self.layer_1_a = Inc(in_channels=1, filters=32)
        self.layer_1_b = Inc(in_channels=1, filters=32)

        # 分别处理 a、b 通道的通道注意力模块
        self.layer_2_a = CAM(128, 4)
        self.layer_2_b = CAM(128, 4)

        # 多层 FABlock 模块提取特征（f5 即为最终增强后的 ab 特征）
        self.fa = FABlock(default_conv, 2 * chn_tmp, 3)  # FABlock模块

        # 下采样部分
        self.conv_chndw1 = nn.Conv2d(2 * chn_tmp, chn_tmp, 3, padding=(3 // 2))  # 第一层卷积
        self.dw_act1 = nn.ReLU(inplace=True)  # ReLU激活函数
        self.conv_chndw2 = nn.Conv2d(chn_tmp, self.output_chn, 3, padding=(3 // 2))  # 第二层卷积
        self.dw_act2 = nn.Tanh()  # Tanh激活函数，将输出限制在[-1, 1]范围内

        # ---------------------------L 通道处理模块---------------------------
        # 对 L 通道增强处理 (对应论文亮度结构分支)
        self.l_branch = L_Branch(in_nc=1, out_nc=1, nc=64, bias=True)

        # 对应论文中的多特征竞争融合模块
        self.ccfm = CCFM(128)

        # ---------------------------基于小波的低频风格抑制模块 (WBL-Low)---------------------------
        # 命名贴合论文图 4.3 的 WBL-Low
        self.wbl_low = FeatureExtractDWT()

    def forward(self, input):
        """
        输入:
          input: Lab 图像，形状 [B, 3, H, W] （第 1 通道为 L，其余为 ab）
        输出:
          gr_feature: 全局过渡特征 (GR feature)
          old_output: 空间域增强后的中间输出，直接拼接增强后的 L 通道与 RGB/ab 融合后的颜色特征
          new_output: 经过频段分治与小波重构后的最终全局校正图像 (I_GR)
        """
        # -------------------------------------Lab 分支处理-------------------------------------
        # 提取Lab颜色空间中L和ab通道
        input_L = input[:, 0:1, :, :]  # L通道
        input_ab = input[:, 1:3, :, :]  # ab通道
        input_a = input[:, 1:2, :, :]  # 取索引 1 的那一维，shape = [B, 1, H, W]
        input_b = input[:, 2:3, :, :]  # 取索引 2 的那一维，shape = [B, 1, H, W]

        # 处理L通道，利用 L_Branch 增强亮度信息
        output_L = self.l_branch(input_L)  # [B,64,H,W]

        # 分别通过 Inception 模块
        layer_1_a = self.layer_1_a(input_a)
        layer_1_b = self.layer_1_b(input_b)

        # 分别通过通道注意力模块
        layer_2_a = self.layer_2_a(layer_1_a)
        layer_2_b = self.layer_2_b(layer_1_b)

        # 通过 CCFM 模块竞争融合
        cat = self.ccfm(layer_2_a, layer_2_b)

        # 经过多个特征注意力 FABlock 模块，直接得到增强后的 ab 分支输出 f5
        f1 = self.fa(cat)
        f2 = self.fa(f1)
        f3 = self.fa(f2)
        f4 = self.fa(f3)
        f5 = self.fa(f4)  # f5 即为所需的增强 ab 特征，形状预计为 [B,128,H,W]

        # -------------------------------------下采样部分-------------------------------------
        dw1 = self.conv_chndw1(f5)
        dw1_nl = self.dw_act1(dw1)  # ReLU激活
        gr_feature = dw1_nl  # 全局校正特征 (GR feature)
        dw2 = self.conv_chndw2(dw1_nl)  # 第二层卷积，dw2是 ∆(Iab)

        # 残差连接
        output_ab = input_ab + dw2  # 将初始的输入ab通道与校正结果相加
        output_ab = self.dw_act2(output_ab)  # Tanh激活

        # -------------------------------------最终融合-------------------------------------
        # 将增强后的 L 通道与颜色特征直接拼接
        old_output = torch.cat((output_L, output_ab), dim=1)  # [B,3,H,W]

        # 经过 WBL-Low 小波模块进行低频退化剥离
        new_output = self.wbl_low(old_output)

        # 返回全局校正特征、空间中间结果、频域最终输出 (I_GR)
        return gr_feature, old_output, new_output  #cc_feature再经过一层下采样卷积变成∆(Iab)，new_output是ICC
