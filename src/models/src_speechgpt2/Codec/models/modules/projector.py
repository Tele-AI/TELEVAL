#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Projector modules."""

import torch

from .conv_layers import NonCausalConv1d, CausalConv1d


class Projector(torch.nn.Module):
    def __init__(self,
        mode,
        input_channels,
        output_channels, 
        stride=1,
        bias=False,
        model='conv1d',
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")

        if model == 'conv1d':
            self.project = Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=bias)
        elif model == 'conv1d_bn':
            self.project = torch.nn.Sequential(
                Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=bias),
                torch.nn.BatchNorm1d(output_channels)
            )
        else:
            raise NotImplementedError(f"Model ({model}) is not supported!")
        
    def forward(self, x): 
        return self.project(x)
    
    def encode(self, x):
        return self.project.inference(x)
