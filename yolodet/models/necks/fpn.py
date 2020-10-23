#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/10 14:16
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: fpn
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolodet.models.backbones.base import DarknetConv2D_Norm_Activation
from yolodet.models.necks.base import SPP, CoordConv, DropBlock2D
from yolodet.models.utils.torch_utils import initialize_weights


class Upsample_Module(nn.Module):
    def __init__(self,in_channels,norm_type,num_groups):
        super(Upsample_Module,self).__init__()
        self.c_conv_1x1 = CoordConv(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1,
                               norm_type=norm_type, num_groups=num_groups)
    def forward(self, x):
        x = self.c_conv_1x1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

'''
论文中C5->P5(input:512,ouput:512)
论文中C4->P4(input:512,ouput:256)
论文中C3->P3(input:256,ouput:128)
实际代码实现中因为resnet 输出[512,1024,2048]
代码实现：C5->P5(input:2048,ouput:512)
代码实现：C4->P4(input:1024,ouput:256)
代码实现：C3->P3(input:512,ouput:128)
最新的paddle代码实现对于C5-->P5 DropBlock的位置和论文中给出的位置不一致。论文中是在第一个Conv Block中采用DropBlock，但代码中是在第二个Conv Block块中采用DropBlock，我们以实际代码实现为主
'''

class PPFPN(nn.Module):
    def __init__(self, in_channels = [512,1024,2048],coord_conv=True,drop_block=True,spp=True,spp_kernel_sizes = [5, 9, 13],block_size=3,keep_prob=0.9,norm_type='BN',num_groups=None,second_drop_block=True):
        super(PPFPN,self).__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.coord_conv = coord_conv
        self.coord_conv = coord_conv
        self.drop_block = drop_block
        self.block_size = block_size
        self.spp = spp
        self.spp_kernel_sizes = spp_kernel_sizes
        self.keep_prob = keep_prob
        # self.second_drop_block = second_drop_block

        #最新的paddle代码实现对于C5-->P5 DropBlock的位置和论文中给出的位置不一致。论文中是在第一个Conv Block中采用DropBlock，但代码中是在第二个Conv Block块中采用DropBlock，我们以实际代码实现为主
        self.p5_layer = self.make_conv_block(in_channels=in_channels[-1],out_channels= in_channels[-1]//2**2,spp=self.spp,second_drop_block=second_drop_block)
        self.p5_upsample = Upsample_Module(in_channels=in_channels[-1]//2**2,norm_type=self.norm_type,num_groups=self.num_groups)
        self.p4_layer = self.make_conv_block(in_channels=in_channels[-2]+in_channels[-1]//2**3,out_channels= in_channels[-2]//2**2,spp=False)
        self.p4_upsample = Upsample_Module(in_channels=in_channels[-2]//2**2,norm_type=self.norm_type,num_groups=self.num_groups)
        self.p3_layer = self.make_conv_block(in_channels=in_channels[-3]+in_channels[-2]//2**3,out_channels= in_channels[-3]//2**2,spp=False)

        self.p5_conv_block_layers = []
        for i, conv in enumerate(self.p5_layer):
            layer_name = 'p5_conv_block_layers_{}'.format(i + 1)
            self.add_module(layer_name, conv)
            self.p5_conv_block_layers.append(layer_name)

        self.p3_conv_block_layers = []
        for i, conv in enumerate(self.p3_layer):
            layer_name = 'p3_conv_block_layers_{}'.format(i + 1)
            self.add_module(layer_name, conv)
            self.p3_conv_block_layers.append(layer_name)

        self.p4_conv_block_layers = []
        for i, conv in enumerate(self.p4_layer):
            layer_name = 'p4_conv_block_layers_{}'.format(i + 1)
            self.add_module(layer_name, conv)
            self.p4_conv_block_layers.append(layer_name)

    def forward(self, input):
        out_5 = input[-1]
        out_4 = input[-2]
        out_3 = input[-3]
        out = []
        for i, layer_name in enumerate(self.p5_conv_block_layers):
            conv_block = getattr(self, layer_name)
            out_5 = conv_block(out_5)
        out.append(out_5)
        out_5 = self.p5_upsample(out_5)
        out_4 = torch.cat([out_5, out_4], dim=1)
        for i, layer_name in enumerate(self.p4_conv_block_layers):
            conv_block = getattr(self, layer_name)
            out_4 = conv_block(out_4)
        out.append(out_4)
        out_4 = self.p4_upsample(out_4)
        out_3 = torch.cat([out_4, out_3], dim=1)
        for i, layer_name in enumerate(self.p3_conv_block_layers):
            conv_block = getattr(self, layer_name)
            out_3 = conv_block(out_3)
        out.append(out_3)

        return out

    def make_conv_block(self,in_channels,out_channels,spp=False,second_drop_block=False):

        conv_layer = []
        c_conv1x1 = CoordConv(in_channels=in_channels, out_channels=out_channels,kernel_size=1,norm_type=self.norm_type,num_groups=self.num_groups,coord_conv=self.coord_conv)
        conv_layer.append(c_conv1x1)

        for j in range(2):
            if j==0:
                # conv Block 1
                conv3x3_1 = DarknetConv2D_Norm_Activation(out_channels, out_channels * 2, kernel_size=3,
                                                           activation='leaky', norm_type=self.norm_type,
                                                           num_groups=self.num_groups)
                if not second_drop_block and self.drop_block:
                    dropblock = DropBlock2D(block_size=self.block_size, keep_prob=self.keep_prob)

                c_conv1x1_1 = CoordConv(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1,
                                         norm_type=self.norm_type, num_groups=self.num_groups,coord_conv=self.coord_conv)
                conv_layer.append(conv3x3_1)
                if not second_drop_block:
                    conv_layer.append(dropblock)
                conv_layer.append(c_conv1x1_1)

                if spp:#spp
                    _spp_ = SPP(self.spp_kernel_sizes)
                    conv1x1 = DarknetConv2D_Norm_Activation(out_channels * 4, out_channels, kernel_size=1,
                                                             activation='leaky',
                                                             norm_type=self.norm_type, num_groups=self.num_groups)
                    conv_layer.append(_spp_)
                    conv_layer.append(conv1x1)
            else:
                # conv Block 2
                conv3x3_2 = DarknetConv2D_Norm_Activation(out_channels, out_channels * 2, kernel_size=3,
                                                           activation='leaky',
                                                           norm_type=self.norm_type, num_groups=self.num_groups)
                if second_drop_block and self.drop_block:
                    dropblock = DropBlock2D(block_size=self.block_size, keep_prob=self.keep_prob)
                c_conv1x1_2 = CoordConv(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1,
                                         norm_type=self.norm_type, num_groups=self.num_groups,coord_conv=self.coord_conv)

                conv_layer.append(conv3x3_2)
                if second_drop_block:
                    conv_layer.append(dropblock)
                conv_layer.append(c_conv1x1_2)
        return conv_layer

    def init_weights(self):
        initialize_weights(self)
