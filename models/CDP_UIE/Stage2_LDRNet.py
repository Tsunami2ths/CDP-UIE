from models.CDP_UIE.BaseNet import BaseNet
from models.CDP_UIE.Stage2_LDR.LDRNet import LDRNet  # 导入真正的第二阶段核心模块

class Stage2_LDRNet(BaseNet):
    def __init__(self, opt):
        super(Stage2_LDRNet, self).__init__()
        self.ldr_module = LDRNet()
        self.init_net(self, opt.init_type, opt.init_gain, opt.gpu_ids)

    def forward(self, input):
        # 直接通过第二阶段核心模块提取高频细节与最终重构结果
        ldr_feature, ldr_output = self.ldr_module(input)

        return ldr_feature, ldr_output