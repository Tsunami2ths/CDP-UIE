import torch
from collections import OrderedDict
from .base_model import BaseModel
from .CDP_UIE.Stage2_LDRNet import Stage2_LDRNet
from .CDP_UIE.Losses.Stage2_LDRLoss import LDRLoss
from .CDP_UIE.Public.util.LAB2RGB_v2 import Lab2RGB

class Stage2_Model(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.netG_LDR = Stage2_LDRNet(opt)
        self.lab2rgb = Lab2RGB()

        if self.isTrain:
            self.loss_G = LDRLoss()
            self.optimizer_G = torch.optim.Adam(
                self.netG_LDR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.loss_names = ["all"]
            self.visual_names = ['raw_rgb', 'ref_rgb', 'pred_ldr']
        else:
            self.visual_names = ['pred_ldr']
        self.model_names = ['G_LDR']

    def set_input(self, input):
        self.input = input
        self.raw = input['raw'].to(self.device)
        if self.isTrain:
            self.ref = input['ref'].to(self.device)
        self.image_paths = input['raw_paths']

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()
            self.optimizer_G.zero_grad()
            self.__backward()
            for group in self.optimizer_G.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)
            self.optimizer_G.step()

    def forward(self):
        self.pred_ldr_feature, self.pred_ldr_raw = self.netG_LDR(self.raw)

        if self.isTrain:
            self.raw_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
            self.ref_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref)

        # 假设 Stage2 输出是 LAB，需要转 RGB
        self.pred_ldr = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_ldr_raw)

    def __backward(self):
        if self.isTrain:
            self.loss_all = self.loss_G(self.raw, self.pred_ldr_raw, self.ref)

            loss_ldr = self.loss_G.get_losses()

            if isinstance(loss_ldr, OrderedDict):
                for k, v in loss_ldr.items():
                    self.loss_names.append(k)
                    setattr(self, "loss_" + k, v)

            self.loss_all.backward()