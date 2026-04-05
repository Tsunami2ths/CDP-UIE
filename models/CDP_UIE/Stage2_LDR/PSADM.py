import torch.nn as nn
import torch



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


class PSADM(nn.Module):
    def __init__(self):
        super(PSADM, self).__init__()
        # 分别处理 R、G、B 通道的 Inception 模块
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)

        # 分别处理 R、G、B 通道的通道注意力模块
        self.layer_2_r = CAM(256, 4)
        self.layer_2_g = CAM(256, 4)
        self.layer_2_b = CAM(256, 4)

        # 拼接后的 Inception 模块
        self.layer_3 = Inc(768, 64)

        # # 添加 DWT_transform 模块
        # self.dwt = DWT_transform(256)

        self.layer_4 = CAM(256, 4)



    def forward(self, input):
        # 将输入的 RGB 图像拆分为 R、G、B 三个通道
        input_r = torch.unsqueeze(input[:, 0, :, :], dim=1)
        input_g = torch.unsqueeze(input[:, 1, :, :], dim=1)
        input_b = torch.unsqueeze(input[:, 2, :, :], dim=1)

        # 分别通过 Inception 模块
        layer_1_r = self.layer_1_r(input_r)
        layer_1_g = self.layer_1_g(input_g)
        layer_1_b = self.layer_1_b(input_b)

        # 分别通过通道注意力模块
        layer_2_r = self.layer_2_r(layer_1_r)
        layer_2_g = self.layer_2_g(layer_1_g)
        layer_2_b = self.layer_2_b(layer_1_b)

        # 将 R、G、B 三个通道的输出在通道维度上拼接
        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)

        # # 通过拼接后的 Inception 模块
        # layer_3 = self.layer_3(layer_concat)
        # output = self.layer_4(layer_3)

        output = layer_concat


        return output
