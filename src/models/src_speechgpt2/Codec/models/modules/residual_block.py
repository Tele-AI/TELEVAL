#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Residual block modules."""

import torch
import torch.nn as nn

from .conv_layers import NonCausalConv1d, CausalConv1d   

class HiFiGANResidualBlock(nn.Module):
    """Causal Residual block module in HiFiGAN."""

    def __init__(
        self,
        mode,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1}
    ):
        """Initialize CausalResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            groups (int): The group number of conv1d (default: 1)
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        for dilation in dilations:
            self.convs1 += [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        groups=groups,
                        bias=bias,
                    )
                ]
        self.num_layer = len(self.convs1)
    
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(self.num_layer):
            xt = self.convs1[idx](self.activation(x))
            if self.use_additional_convs:
                xt = self.convs2[idx](self.activation(xt))
            x = xt + x
        return x
    
    def inference(self, x):
        for idx in range(self.num_layer):
            xt = self.convs1[idx].inference(self.activation(x))
            if self.use_additional_convs:
                xt = self.convs2[idx].inference(self.activation(xt))
            x = xt + x
        return x
    