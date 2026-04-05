from models.CDP_UIE.BaseNet import BaseNet
from models.CDP_UIE.Stage1_GR.GRNet import GRNet


class Stage1_GRNet(BaseNet):
    def __init__(self, opt):
        super(Stage1_GRNet, self).__init__()
        self.gr_module = GRNet()
        self.init_net(self, opt.init_type, opt.init_gain, opt.gpu_ids)

    def forward(self, input):
        gr_feature, gr_old_output, gr_new_output = self.gr_module(input)
        return gr_feature, gr_old_output, gr_new_output

    def disable_grad(self):
        # 冻结第一阶段核心模块的梯度
        for name, param in self.gr_module.named_parameters():
            param.requires_grad = False

