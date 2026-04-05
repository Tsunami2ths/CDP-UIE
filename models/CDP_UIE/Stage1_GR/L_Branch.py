import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义通道注意力模块 (Channel Attention Block)
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力计算模块
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),  # 恢复维度
            nn.Sigmoid()  # 激活函数，输出权重
        )

    def forward(self, x):
        # 计算通道注意力权重
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # 将权重与输入相乘
        return x * y




# 定义混合膨胀残差注意力模块
class MRARB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(MRARB, self).__init__()
        kernel_size = 3
        reduction = 8

        # 通道注意力模块
        self.cab = CAB(in_channels, reduction, bias)

        # 定义多个卷积层，使用不同的膨胀率
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

    def forward(self, y):
        # 多尺度卷积 + 残差连接
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1 + y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1 + y1)
        y8_1 = y8 + y6_1

        # 通道注意力
        y9 = self.cab(y8_1)
        y9_1 = y + y9

        return y9_1


class L_Branch(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64,bias=True): #输入特征图的通道数 输出特征图的通道数 中间特征图的通道数 是否在卷积层中使用偏置项
        super(L_Branch, self).__init__()
        kernel_size = 3


        # 头部卷积层  conv_head 的主要作用是将输入特征图的通道数从 in_nc 调整为 nc。
        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.mrarb = MRARB(nc, nc, bias)

        # 尾部卷积层
        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)



    def forward(self, x): #[6, 3, 256, 256]
        """
        x shape: torch.Size([6, 3, 256, 256])
        x1 shape: torch.Size([6, 128, 256, 256])
        x6 shape: torch.Size([6, 128, 256, 256])
        x7 shape: torch.Size([6, 3, 256, 256])
        X shape: torch.Size([6, 3, 256, 256])
        z shape: torch.Size([6, 3, 256, 256])
        Z shape: torch.Size([6, 3, 256, 256])
        Output shape: torch.Size([6, 3, 256, 256])

        """

        # 跳跃连接 HDRAB 模块
        y1 = self.conv_head(x)  # [6, nc, 256, 256]

        y2 = self.hdrab(y1)  # [6, nc, 256, 256]
        y3 = self.hdrab(y2)
        y4 = self.hdrab(y3)
        y5 = self.hdrab(y4 + y3)
        y6 = self.hdrab(y5 + y2)  #y5 + y2是逐元素相加
        y7 = self.conv_tail(y6 + y1)
        Y = x - y7  # 第二路输出

        # Y = y6 + y1
        return Y
