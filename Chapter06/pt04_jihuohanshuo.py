#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:07:08 2018

@author: pc
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
m = F.tanh()
inputS = autograd.Variable(torch.randn(2))
print(inputS)
print(m(inputS))