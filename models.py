import torch
import torch.nn as nn
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.sage_conv import SAGEConv

from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        """两个简单的线性层"""
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, out_feats)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.softmax(x, dim=-1)
