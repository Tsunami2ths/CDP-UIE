import torch
import torch.nn as nn
import torch.nn.functional as F
# from wavelet import wt, iwt


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
        y = self.avg_pool(x)  # 全局最大池化
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 1D卷积
        y = self.sigmoid(y)  # Sigmoid激活
        return x * y.expand_as(x)  # 应用通道注意力



# ----------------------- DWT_transform 模块 ---------------------------
class DWT_transform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT()  # DWT模块

        # 定义多个卷积和归一化层
        self.res_l = ConvINRe(in_channels)
        self.res_l2 = ConvINRe(in_channels)
        self.res_h1 = ConvBNRe(in_channels)
        self.res_h2 = ConvBNRe(in_channels)
        self.res_h3 = ConvBNRe(in_channels)


        # 定义多个通道注意力层
        self.pa_low = PALayer_l(in_channels)
        self.ca_low = CALayer_low(in_channels)
        self.pa_high1 = PALayer_h(in_channels)
        self.ca_high1 = CALayer_high(in_channels)
        self.pa_high2 = PALayer_h(in_channels)
        self.ca_high2 = CALayer_high(in_channels)
        self.pa_high3 = PALayer_h(in_channels)
        self.ca_high3 = CALayer_high(in_channels)


        # 定义均值和标准差的卷积层
        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1, 2)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1, 2)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)

        # 权重生成模块，用于融合低频和高频
        self.weight_conv = ConvLayer(in_channels, 1, kernel_size=3, stride=1)

    def forward(self, x):
        LL, HL, LH, HH = self.dwt(x)  # 执行DWT

        # 对低频分量进行处理
        LL, mean, std = PONO(LL)  # 对低频分量进行PONO
        mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))  # 计算均值
        std = self.std_conv3(self.std_conv2(self.std_conv1(std)))  # 计算标准差
        LL = self.res_l(LL)  # 低频分量的卷积
        LL = self.res_l2(LL)  # 低频分量的卷积
        LL_ca = self.ca_low(LL)  # 低频分量的通道注意力
        LL_pa = self.pa_low(LL_ca)  # 低频分量的空间注意力
        LL_pa = MS(LL_pa, mean, std)  # 反归一化


        # 使用IWT恢复到原始分辨率(N,C,H,W)
        enhanced_feature = iwt((LL_pa, HL, LH, HH))  # IWT反变换

        return enhanced_feature


# ----------------------- 运行示例 ---------------------------
if __name__ == '__main__':
    # 模拟输入（假设水下图像的特征表示），尺寸为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 256, 256)
    print("输入张量形状:", input_tensor.shape)

    # 实例化模型
    model = DWT_transform(in_channels=64)
    output = model(input_tensor)
    print("增强后特征图形状:", output.shape)