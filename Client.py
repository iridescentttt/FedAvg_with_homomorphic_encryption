from os import X_OK
from numpy.lib.function_base import gradient
from collections import defaultdict
from models import *
import numpy as np
import torch.nn as nn
from torch import autograd
from sklearn.metrics import f1_score


class Client(nn.Module):
    def __init__(self, train_loader, mode, in_feats, h_feats, num_classes, lr, device):
        """定义client"""
        super(Client, self).__init__()
        self.train_loader = train_loader
        self.mode = mode
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.device = device
        self.prox_miu = 1.0

        self.model = Model(in_feats, h_feats, num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        

    def supervisedTrain(self):
        """进行训练"""
        self.train()

        total_examples = total_loss = 0
        for batch in self.train_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            out = self.model(x)
            loss = F.cross_entropy(out, y)

            loss.backward()
            self.optimizer.step()

            total_examples += len(x)
            total_loss += float(loss) * len(x)
        return total_loss / total_examples

    @torch.no_grad()
    def test(self, test_loader):
        """进行测试"""
        self.eval()
        pred_all = []
        y_all = []

        for batch in test_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            out = self.model(x)
            pred = out.argmax(dim=-1)

            pred_all += pred.tolist()
            y_all += y.tolist()
            
        micro_f1 = f1_score(y_all, pred_all, average='micro')
        return micro_f1
