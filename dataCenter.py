import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import *


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x.float()
        self.y = y.long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class DataCenter(object):

    def __init__(self, client_num, device):
        """dataCenter参数"""
        super(DataCenter, self).__init__()
        self.client_num = client_num
        self.device = device

    def load_dataSet(self, batch_size):
        """划分数据集"""
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(
            './data/', train=True, transform=transformation, download=True)
        test_dataset = datasets.MNIST(
            './data/', train=False, transform=transformation, download=True)
        test_data = MyDataset(train_dataset.data, train_dataset.targets)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True)

        train_loader = []

        """进行non-iid的划分"""
        users_dict = mnist_noniid(train_dataset, self.client_num)
        for cid in range(self.client_num):
            client_data = MyDataset(
                train_dataset.data[users_dict[cid]], train_dataset.targets[users_dict[cid]])
            # client_data = MyDataset(train_dataset.data, train_dataset.targets)
            loader = DataLoader(
                client_data, batch_size=batch_size, shuffle=True)
            train_loader.append(loader)
        setattr(self, "train_loader", train_loader)
        setattr(self, "test_loader", test_loader)
        setattr(self, "in_feats",
                train_dataset.data[0].shape[0]*train_dataset.data[0].shape[1])
        setattr(self, "num_classes", len(set(train_dataset.targets.numpy())))
