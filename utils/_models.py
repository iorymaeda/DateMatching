import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=32, d=[64, 64], **kwargs):
        super().__init__()
        
        seq = []
        d = [input_dim] + d + [1]
        for i in range(len(d)-1):
            seq.append(
                nn.Linear(d[i], d[i+1])
            )
            seq.append(nn.Dropout(p=0.5))
            if i != len(d)-2:
                seq.append(nn.GELU())
                
        self.seq = nn.Sequential(*seq)

        
    def forward(self, x: torch.Tensor):
        return self.seq(x)
        