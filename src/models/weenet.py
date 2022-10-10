#!/usr/bin/env python
import torch.nn as nn
import torch.nn.functional as F


class WeeNet(nn.Module):
    def __init__(self):
        super(WeeNet, self).__init__()
        self.layer1 = nn.Conv2d(
            3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.layer2 = nn.Conv2d(
            32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.layer3 = nn.Conv2d(
            64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))

        return out


class DenseWeeNet(nn.Module):
    def __init__(self, D_in, D_out):
        super(DenseWeeNet, self).__init__()
        self.layer1 = nn.Linear(D_in, D_out, bias=True)

    def forward(self, x):
        out = F.relu(self.layer1(x))

        return out


class Conv2dWeeNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv2dWeeNet, self).__init__()
        self.layer1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), bias=True)

    def forward(self, x):
        out = F.relu(self.layer1(x))

        return out
