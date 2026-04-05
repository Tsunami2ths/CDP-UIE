import torch.nn as nn
from models.CDP_UIE.Stage2_LDR.PSADM import PSADM
from models.CDP_UIE.Stage2_LDR.WBL_High import FeatureExtractDWT_High


class LDRNet(nn.Module):
    def __init__(self,
                 input_chn=3,      # 输入通道数，默认为3（RGB图像）
                 n_feat=256,       # 特征通道数，默认为256
                 output_chn=3,     # 输出通道数，默认为3
                 chan_factor=2):   # 通道因子，用于控制特征通道数的缩放
        super(LDRNet, self).__init__()

        # 初始化参数
        self.input_chn = input_chn
        self.output_chn = output_chn
        self.n_feat = n_feat
        self.chan_factor = chan_factor

        # 核心模块1：物理光谱感知解耦模块 (对应论文4.4.1)
        self.psadm = PSADM()

        # 核心模块2：高频定向增强模块 WBL-High (对应论文4.4.2)
        self.wbl_high = FeatureExtractDWT_High()

        # 初始化网络层
        self._init_layers()

    def _init_layers(self):
        # 初始化卷积层
        self.conv = nn.Sequential(
            # 1×1 卷积将 PSADM 拼接后的 768 通道降至 256，实现跨波段特征融合
            nn.Conv2d(768, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        # 1. 跨域回归：首先进入物理光谱感知解耦模块 (PSADM)
        psadm_out = self.psadm(input)

        # 2. 初始跨通道信息交互与降维 (对应论文中的 1x1 卷积融合)
        ldr_feature = self.conv(psadm_out)


        # 3. 高频纹理定向复原：进入 WBL-High 模块进行稀疏高频提纯
        out = self.wbl_high(ldr_feature)  # 最终输出 torch.Size([8, 3, 256, 256])

        # 返回局部重构特征和最终输出 (I_out)
        return ldr_feature, out