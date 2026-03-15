import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, d_in:int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(d_in, 1)


    def forward(self, x:torch.Tensor):
        return self.linear(x)