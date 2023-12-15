#!/usr/bin/env python
# coding: utf-8
"""
Linear regression
"""

import torch
from torch import nn

# build linear model
class linearRegression(nn.Module):

    def __init__(self, in_size, out_size):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out
