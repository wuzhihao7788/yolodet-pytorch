#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: base.py
# @Software: PyCharm
# @github ：https://github.com/wuzhihao7788/yolodet-pytorch

               ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃              ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━-┓
                ┃Beast god bless┣┓
                ┃　Never BUG ！ ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
=================================================='''

import torch.nn as nn
import torch

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class DarknetConv2D_Norm_Activation(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, activation='mish', norm_type='BN',num_groups=32, bias=False):
        super(DarknetConv2D_Norm_Activation, self).__init__()
        pad = (kernel_size-1)//2 # kernel size is 3 or 0
        self.sigmoid = False
        assert norm_type in (None,'BN','GN'),'norm type just support BN or GN'
        self.norm_type = norm_type
        self.darknetConv = nn.ModuleList()
        self.darknetConv.add_module('conv',nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=pad,bias=bias))
        if norm_type == 'BN':
            self.darknetConv.add_module('bn',nn.BatchNorm2d(out_channels))
        elif norm_type == 'GN':
            if not isinstance(num_groups,int):
                num_groups = 32
            self.darknetConv.add_module('gn',nn.GroupNorm(out_channels,num_groups=num_groups))
        if activation == 'relu':
            self.darknetConv.add_module('relu',nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.darknetConv.add_module('leaky',nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'swish':
            self.darknetConv.add_module('swish',Swish())
        elif activation == 'mish':
            self.darknetConv.add_module('mish',Mish())
        elif activation == 'logistic':
            self.darknetConv.add_module('logistic',torch.nn.Sigmoid())
        else:
            pass

    def forward(self, x):
        for dc in self.darknetConv:
            x = dc(x)
        return x

#base res unit
class ResBlock(nn.Module):

    def __init__(self,in_channels,out_channels,norm_type='BN',num_groups=None):
        super(ResBlock,self).__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.conv1x1 = DarknetConv2D_Norm_Activation(in_channels, out_channels, kernel_size=1, activation='mish',norm_type=self.norm_type,num_groups=self.num_groups)
        self.conv3x3 = DarknetConv2D_Norm_Activation(out_channels, in_channels, kernel_size=3, activation='mish',norm_type=self.norm_type,num_groups=self.num_groups)


    def forward(self,x):
        identity = x
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        out += identity
        return out


#csp resbock body
class CSP_ResBock_Body(nn.Module):
    def __init__(self,in_channels,res_num,norm_type='BN',num_groups=None):
        super(CSP_ResBock_Body,self).__init__()
        self.in_channels = in_channels
        self.res_num = res_num
        self.norm_type = norm_type
        self.num_groups = num_groups

        self.downsample = DarknetConv2D_Norm_Activation(self.in_channels, self.in_channels * 2, kernel_size=3, stride=2,norm_type=self.norm_type,num_groups=self.num_groups)
        if res_num == 1:
            #in_channels = 32
            self.conv1x1_1 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels * 2, kernel_size=1, stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            resBlocks = self.make_res_blocks(channels=self.in_channels*2, res_num=self.res_num)
            self.conv1x1_4 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels * 2, kernel_size=1, stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            self.conv1x1_2 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels * 2, kernel_size=1,
                                                           stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            self.conv1x1_3 = DarknetConv2D_Norm_Activation(self.in_channels * 2 * 2, self.in_channels * 2, kernel_size=1,
                                                           stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
        else:
            #in_channels = 64
            self.conv1x1_1 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels, kernel_size=1, stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            self.conv1x1_4 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels, kernel_size=1, stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            self.conv1x1_2 = DarknetConv2D_Norm_Activation(self.in_channels, self.in_channels, kernel_size=1,
                                                           stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            self.conv1x1_3 = DarknetConv2D_Norm_Activation(self.in_channels * 2, self.in_channels * 2, kernel_size=1,
                                                           stride=1,norm_type=self.norm_type,num_groups=self.num_groups)
            resBlocks = self.make_res_blocks(channels=self.in_channels,res_num=self.res_num)
        self.res_block_layers = []
        for i ,resBlock in enumerate(resBlocks):
            layer_name = 'res_block_layer_{}'.format(i + 1)
            self.add_module(layer_name,resBlock)
            self.res_block_layers.append(layer_name)

    def forward(self, x):
        out = self.downsample(x)
        identity = out
        out = self.conv1x1_1(out)
        for res_name in self.res_block_layers:
            res_block = getattr(self,res_name)
            out = res_block(out)

        out = self.conv1x1_2(out)
        out1 = self.conv1x1_4(identity)
        out = torch.cat([out, out1], dim=1)

        out = self.conv1x1_3(out)

        return out

    def make_res_blocks(self,channels,res_num):
        res_blocks = []
        if res_num == 1:
            res_blocks.append(ResBlock(channels,int(channels/2),norm_type=self.norm_type,num_groups=self.num_groups))
        else:
            for i in range(res_num):
                res_blocks.append(ResBlock(channels, channels,norm_type=self.norm_type,num_groups=self.num_groups))
        return nn.Sequential(*res_blocks)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,in_channels=3,out_channels=80, kernel_size=3, stride=1,activation='leaky',norm_type='BN',num_groups=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = DarknetConv2D_Norm_Activation(in_channels * 4, out_channels, kernel_size=kernel_size, stride=stride,activation=activation,norm_type=norm_type,num_groups=num_groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,activation='leaky',norm_type='BN',num_groups=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels 80
        self.cv1 = DarknetConv2D_Norm_Activation(c1, c_, 1, 1,activation=activation,norm_type=norm_type,num_groups=num_groups) # 160 80
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)# 160 80
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)# 80 80
        self.cv4 = DarknetConv2D_Norm_Activation(2 * c_, c2, 1, 1,activation=activation,norm_type=norm_type,num_groups=num_groups) # 160 160
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3) 160
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])#80 80 True 1

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True,e=0.5,activation='leaky',norm_type='BN',num_groups=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DarknetConv2D_Norm_Activation(c1, c_, 1, 1,activation=activation,norm_type=norm_type,num_groups=num_groups)
        self.cv2 = DarknetConv2D_Norm_Activation(c_, c2, 3, 1,activation=activation,norm_type=norm_type,num_groups=num_groups)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))






