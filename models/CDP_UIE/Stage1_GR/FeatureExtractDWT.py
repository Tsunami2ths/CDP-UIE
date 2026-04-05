import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CDP_UIE.Stage1_GR.WBL_Low import DWT_transform

# 定义一个卷积层模块，用于反射填充后卷积
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

# 全局颜色注意力模块（同时考虑光照和颜色信息）
class GlobalColorAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 全局光照注意力
        self.light_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // 8, 1),  # 压缩通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        # 颜色分布注意力
        self.color_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算光照注意力（输出形状 [B, in_channels, 1, 1]）
        light_att = self.light_att(x)
        # 计算颜色注意力（输出形状同 x）
        color_att = self.color_att(x)
        # 结合两者，并将注意力应用于输入特征
        att = light_att * color_att
        return x * att

# 尾部卷积模块，将高维特征转换为最终的 RGB 输出
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

# 修改后的 FeatureExtractDWT 模块
# 输入通道为192，先降维到64，再进入后续模块处理
class FeatureExtractDWT(nn.Module):
    def __init__(self, in_ch=3, mid_ch=64):
        super().__init__()
        # 1x1 卷积用于降维：将128维降至64维
        # self.reduce_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.reduce_conv = ConvLayer(in_ch, mid_ch, kernel_size=3, stride=1)
        # 使用 DWT_transform，输入和输出通道均为64
        self.dwt_transform = DWT_transform(mid_ch)
        # 全局颜色注意力模块，输入通道64
        self.global_color_att = GlobalColorAttention(in_channels=mid_ch)
        # 尾部卷积层，将64通道转换为3通道 RGB 输出
        self.tail = TailConv(in_channels=mid_ch, mid_channels=32, out_channels=3)

    def forward(self, lab):
        # Step 1: 降维，将 192 维输入降为 64 维
        reduced = self.reduce_conv(lab)  # [B, 64, H, W]
        # Step 2: 使用 DWT_transform 处理
        enhanced = self.dwt_transform(reduced)  # 依 DWT_transform 实现，输出 [B, 64, H, W]
        # Step 3: 通过全局颜色注意力模块
        enhanced = self.global_color_att(enhanced)  # [B, 64, H, W]
        # Step 4: 通过尾部卷积层映射为最终 lab 图像
        output = self.tail(enhanced)  # [B, 3, H, W]
        return output

if __name__ == "__main__":
    # 示例测试代码
    # 假设输入为一个随机张量，形状为 (batch_size, 192, height, width)
    input_tensor = torch.randn(8, 3, 256, 256)
    model = FeatureExtractDWT(in_ch=3, mid_ch=64)
    output = model(input_tensor)
    print("Output shape:", output.shape)

