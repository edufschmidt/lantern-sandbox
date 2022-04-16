import torch
from torch import nn

import numpy as np


class ReLU(nn.ReLU):
    r"""ReLU activation with weight initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def init_weights_fcn(self, m):
        if type(m) == nn.Linear:
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')        

    @torch.no_grad()
    def init_first_layer_weights_fcn(self, m):
        pass


class Sine(nn.Module):
    r"""Sine activation with weight initialization 
    as described in Sitzmann et al. 2020.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def init_weights_fcn(self, m):        
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)            
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    @torch.no_grad()
    def init_first_layer_weights_fcn(self, m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, input):
        return torch.sin(30 * input)
