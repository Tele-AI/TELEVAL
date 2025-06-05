import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .conv_layers import CausalConv1d, CausalConvTranspose1d, NonCausalConv1d, NonCausalConvTranspose1d
from .multi_fusion import MultiReceptiveField, MultiGroupConv1d

class Decoder(nn.Module):
    """HiFiGAN causal generator module."""

    def __init__(
        self,
        mode,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        stats=None,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            groups (int): Number of groups of residual conv
            bias (bool): Whether to add bias parameter in convolution layers.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            stats (str): File name of the statistic file

        """
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
            ConvTranspose1d = NonCausalConvTranspose1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
            ConvTranspose1d = CausalConvTranspose1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # Group conv or MRF
        if (len(resblock_dilations) == len(resblock_kernel_sizes) == 1) and (groups > 1):
            multi_fusion = MultiGroupConv1d
        else:
            multi_fusion = MultiReceptiveField
        
        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.input_conv = Conv1d(
            in_channels,
            channels,
            kernel_size,
            stride=1,
        )
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.activation_upsamples = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                ConvTranspose1d(
                    channels // (2 ** i),
                    channels // (2 ** (i + 1)),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_scales[i],
                )
            ]
            self.blocks += [
                multi_fusion(
                    mode=mode,
                    channels=channels // (2 ** (i + 1)),
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilations,
                    groups=groups,
                    bias=bias,
                    use_additional_convs=use_additional_convs,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
        self.activation_output1 = nn.LeakyReLU()
        self.activation_output2 = nn.Tanh()
        self.output_conv = Conv1d(
            channels // (2 ** (i + 1)),
            out_channels,
            kernel_size,
            stride=1,
        )

        # load stats
        if stats is not None:
            self.register_stats(stats)
            self.norm = True
        else:
            self.norm = False
        logging.info(f"Input normalization: {self.norm}")

    
    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        if self.norm:
            c = (c.transpose(2, 1) - self.mean) / self.scale
            c = c.transpose(2, 1)
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](self.activation_upsamples(c))
            c = self.blocks[i](c)
        c = self.output_conv(self.activation_output1(c))
        c = self.activation_output2(c)

        return c
    
    def inference(self, c):
        assert self.mode == "causal"
        if self.norm:
            c = (c.transpose(2, 1) - self.mean) / self.scale
            c = c.transpose(2, 1)
        c = self.input_conv.inference(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i].inference(self.activation_upsamples(c))
            c = self.blocks[i].inference(c)
        c = self.output_conv.inference(self.activation_output1(c))
        c = self.activation_output2(c)

        return c