import math
import copy
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from typing import List, Union, Tuple

from utils import identity

class ConvNormLayer(nn.Module):
    ch_out: int
    kernel_size: Tuple[int, int]
    stride: int
    padding: Union[str, None, List] = None
    bias: bool = False
    act: callable = identity

    def setup(self):
        pad_type = self.padding if self.padding != None else [(self.kernel_size[0] - 1)//2, (self.kernel_size[1] - 1)//2]
        self.conv = nn.Conv(features = self.ch_out, kernel_size = self.kernel_size, strides= self.stride, padding= pad_type, use_bias= self.bias)
        self.norm = nn.BatchNorm()

    def __call__(self, x, train=True):
        return self.act(self.norm(self.conv(x), use_running_average=not train))
    
class RepVggBlock(nn.Module):
    ch_out: int
    act: callable = identity

    def setup(self):
        self.conv1 = ConvNormLayer(self.ch_out, (3,3), 1, padding=[1,1], act = self.act)
        self.conv2 = ConvNormLayer(self.ch_out, (1,1), 1, padding='SAME', act = self.act)
    
    def __call__(self, x, train=True):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)
    
class CSPRepLayer(nn.Module):
    out_channels: int
    num_blocks: int =3,
    expansion: float =1.0,
    bias: bool = False,
    act: callable = jax.nn.silu

    def setup(self):
        hidden_channels = int(self.out_channels * self.expansion)
        self.conv1 = ConvNormLayer(hidden_channels, (1, 1), 1, bias = self.bias, act = self.act)
        self.conv2 = ConvNormLayer(hidden_channels, (1, 1), 1, bias = self.bias, act = self.act)
        self.bottlenecks = nn.Sequential([RepVggBlock(hidden_channels, act = self.act) for _ in range(self.num_blocks)])
        if hidden_channels != self.out_channels:
            self.conv3 = ConvNormLayer(self.out_channels, (1, 1), 1, bias = self.bias, act = self.act)
        else:
            self.conv3 = identity()
    
    def __call__(self, x, train=True):
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)