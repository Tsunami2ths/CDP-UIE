import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph


class CCFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):  #height = 特征分支数
        super(CCFM, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats1, in_feats2):
        in_feats = [in_feats1, in_feats2]   #长度为 2 的 lis，，元素是两个形状均为 [B, C, H, W] 的 Tensor
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)   #四维,[B, 2*C, H, W]
        in_feats = in_feats.view(B, self.height, C, H, W)   #五维,[B, height, C, H, W]

        feats_sum = torch.sum(in_feats, dim=1)  #[B, C, H, W]
        attn = self.mlp(self.avg_pool(feats_sum))  #[B, C, H, W] → [B, C, 1, 1] → [B, height*C, 1, 1]
        attn = self.softmax(attn.view(B, self.height, C, 1, 1)) #[B, height*C, 1, 1] → [B, height, C, 1, 1]

        out = torch.sum(in_feats*attn, dim=1) #[B, C, H, W]
        return out

if __name__ == '__main__':
    block = CCFM(64)
    input1 = torch.rand(8, 64, 256, 256) # 输入 N C H W
    input2 = torch.rand(8, 64, 256, 256)
    output = block(input1, input2)
    print(output.size())
    model_graph = draw_graph(block, input_data=(input1, input2), expand_nested=True)
    model_graph.visual_graph.render("mfm_model", format="png")