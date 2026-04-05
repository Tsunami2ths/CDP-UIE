import torch
from collections import OrderedDict

from .base_model import BaseModel
from .CDP_UIE.Stage1_GRNet import Stage1_GRNet

from .CDP_UIE.Losses.Stage1_GRLoss import GRLoss
from .CDP_UIE.Public.util.LAB2RGB_v2 import Lab2RGB

class Stage1_Model(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.netG_GR = Stage1_GRNet(opt)
        self.lab2rgb = Lab2RGB()

        if self.isTrain:
            # 直接实例化真实的损失函数
            self.loss_G = GRLoss()
            self.optimizer_G = torch.optim.Adam(
                self.netG_GR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.loss_names = ["all"]
            self.visual_names = ['raw_rgb', 'ref_rawLrefab', 'pred_gr', 'ref_rgb', 'pred_gr_refLpredab']
        else:
            self.visual_names = ['pred_gr']
        self.model_names = ['G_GR']

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
        self.pred_gr_feature, self.pred_old_lab, self.pred_new_lab = self.netG_GR(self.raw)

        if self.isTrain:
            ref_L = self.ref[:, 0:1, :, :]
            gr_ab = self.pred_old_lab[:, 1:3, :, :]
            self.pred_gr_refLpredab = torch.cat((ref_L, gr_ab), dim=1)
            raw_L = self.raw[:, 0:1, :, :]
            ref_ab = self.ref[:, 1:3, :, :]
            self.ref_rawLrefab = torch.cat((raw_L, ref_ab), dim=1)
            self.raw_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
            self.ref_rawLrefab = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref_rawLrefab)
            self.ref_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref)
            self.pred_gr_refLpredab = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_gr_refLpredab)

        self.pred_gr = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_new_lab)

    def __backward(self):
        if self.isTrain:
            # 【重大改变】：直接调用真实 Loss 的 forward，只传真正需要的参数
            self.loss_all = self.loss_G(self.raw, self.pred_old_lab, self.pred_new_lab, self.ref)

            # 直接获取底层记录的字典
            loss_gr = self.loss_G.get_losses()

            if isinstance(loss_gr, OrderedDict):
                for k, v in loss_gr.items():
                    self.loss_names.append(k)
                    setattr(self, "loss_" + k, v)

            self.loss_all.backward()