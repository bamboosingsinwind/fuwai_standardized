from onedResnet_branch import resnet18to34
from Transformer_branch import Transformer
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class GDB(nn.Module):
    def __init__(self, num_classes=2):
        super(GDB, self).__init__()
        self.resnet = resnet18to34()
        self.transformer = model = Transformer(d_feature=SIG_LEN//stride, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, dropout=dropout, class_num=class_num)
        
        self.gate = torch.nn.Linear((512+2)+(256+2), 2)
        self.fc = nn.Linear((512+2)+(256+2), num_classes)

    def forward(self, x, feat):
        R = self.resnet(x, feat)
        T = self.transformer(x, feat)

        gate = F.softmax(self.gate(torch.cat([R,T], dim=-1)), dim=-1)
        encoding = torch.cat([R * gate[:, 0:1], T * gate[:, 1:2]], dim=-1)

        output = self.fc(encoding)
        return output

model = GDB(2)    
sig, fea_plus = torch.randn([4,12,5000]),torch.randn([4,2])
pred = model(sig, fea_plus) #
print(pred)
