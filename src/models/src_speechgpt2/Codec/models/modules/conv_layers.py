#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convolution layers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def int2tuple(variable, length):
    if isinstance(variable, int):
        return (variable,)*length
    else:
        assert len(variable) == length, f"The length of {variable} is not {length}!"
        return variable


class Conv1d1x1(nn.Conv1d):
    """1x1 Conv1d."""

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class NonCausalConv1d(nn.Module):
    """1D noncausal convloution w/ 2-sides padding."""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=-1, 
            dilation=1,
            groups=1,
            bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x
    

class NonCausalConvTranspose1d(nn.Module):
    """1D noncausal transpose convloution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride+1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x


class CausalConv1d(NonCausalConv1d):
    """1D causal convloution w/ 1-side padding."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1,
        bias=True,
        pad_buffer=None,
        padding_mode='zeros',
    ):
        super(CausalConv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (kernel_size - 1) * dilation

        self.padding_mode = padding_mode

        self.in_channels = in_channels
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, self.in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)
        
        assert padding_mode == "zeros"
        
    def forward(self, x):
        if self.padding_mode == "zeros":
            pad = nn.ConstantPad1d((self.pad_length, 0), 0.0)
        elif self.padding_mode == "replicate":
            pad = nn.ReplicationPad1d((self.pad_length, 0))
        else:
            assert False
        
        # print(f"before pad x shape = {x.shape}")
        x = pad(x)
        # print(f"after pad x shape = {x.shape}")
        return self.conv(x)
    
    def inference(self, x):
        x = torch.cat((self.pad_buffer, x), -1)
        self.pad_buffer = x[:, :, x.shape[-1] - self.pad_length:]
        return self.conv(x)
    
    def reset_buffer(self, batch_size):
        # self.pad_buffer.zero_()
        device = next(self.parameters()).device
        self.pad_buffer = torch.zeros(batch_size, self.in_channels, self.pad_length, device=device)

    def set_buffer(self, pad_buffer):
        self.pad_buffer = pad_buffer
        
class CausalConvTranspose1d(NonCausalConvTranspose1d):
    """1D causal transpose convloution."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        bias=True,
        pad_buffer=None,
    ):
        super(CausalConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (math.ceil(kernel_size/stride) - 1)
        self.in_channels = in_channels
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, self.in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)
        self.is_buffer_empty = True  # 一开始 buffer 为空
         
    def forward(self, x):
        pad = nn.ReplicationPad1d((self.pad_length, 0))
        # print(f"before pad x shape = {x.shape}")
        x = pad(x)
        # print(f"after pad x shape = {x.shape}")
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def inference(self, x):
        if self.is_buffer_empty: # 如果是第一帧, 则和训练时 forward 一样用最边上的值来 pad
            self.is_buffer_empty = False
            x = nn.ReplicationPad1d((self.pad_length, 0))(x)
        else: # 否则不是第一帧, 那么上一帧已经 pad 过了,用上一帧的结果
            x = torch.cat((self.pad_buffer, x), -1)

        self.pad_buffer = x[:, :, x.shape[-1] - self.pad_length:]
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def reset_buffer(self, batch_size):
        # self.pad_buffer.zero_()
        device = next(self.parameters()).device
        self.pad_buffer = torch.zeros(batch_size, self.in_channels, self.pad_length, device=device)
        self.is_buffer_empty = True
    
    # 禁止在第一帧时调用
    def set_buffer(self, pad_buffer):
        assert pad_buffer != None
        self.pad_buffer = pad_buffer
        self.is_buffer_empty = False

class NonCausalConv2d(nn.Module):
    """2D noncausal convloution w/ 4-sides padding."""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=-1, 
            dilation=1,
            groups=1,
            bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int2tuple(kernel_size, 2)
        self.dilation = int2tuple(dilation, 2)
        if isinstance(padding, int) and padding < 0:
            padding_0 = (self.kernel_size[0] - 1) // 2 * self.dilation[0]
            padding_1 = (self.kernel_size[1] - 1) // 2 * self.dilation[1]
            padding = (padding_0, padding_1)
        
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x
