#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/22 19:54
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: yolov5
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

from yolodet.models.backbones.base import DarknetConv2D_Norm_Activation, BottleneckCSP
from yolodet.models.necks.base import Concat
from yolodet.models.utils.torch_utils import make_divisible, initialize_weights


class YOLOv5FPN(nn.Module):

    def __init__(self,depth_multiple,width_multiple,in_channels = 1024 , out_channels = [512,256,256,512],shortcut = [False,False,False,False],bottle_depths=[3, 3, 3, 3],upsampling_mode = 'nearest',norm_type='BN',num_groups=None):
        super(YOLOv5FPN,self).__init__()
        assert upsampling_mode.lower() in ['nearest','linear','bilinear', 'bicubic','trilinear'],"upsampling mode just support ['nearest','linear','bilinear', 'bicubic','trilinear']"
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.shortcut = shortcut
        self.bottle_depths = bottle_depths
        self.out_channels = out_channels
        c1 = make_divisible(in_channels * self.width_multiple, 8)
        self.concats = []
        self.m = nn.ModuleList()
        for idx,depth in enumerate(bottle_depths):
            _c2 = self.out_channels[idx]
            n = max(round(depth * self.depth_multiple), 1) if depth > 1 else depth  # depth gain
            if idx <2:
                c2 = make_divisible(_c2 * self.width_multiple, 8)
                con1x1 = DarknetConv2D_Norm_Activation(c1, c2, kernel_size=1, stride=1, activation='leaky',norm_type=norm_type, num_groups=num_groups)
                self.m.append(con1x1)
                self.concats.append(len(self.m)-1)
                upsample = nn.Upsample(scale_factor=2, mode=upsampling_mode.lower())
                self.m.append(upsample)
                self.m.append(Concat())
                c1 = c2*2
                bottlenect = BottleneckCSP(c1, c2, n=n, shortcut=self.shortcut[idx])
                self.m.append(bottlenect)
                c1 = c2
            else:
                c2 = make_divisible(_c2 * self.width_multiple, 8)
                con3x3 = DarknetConv2D_Norm_Activation(c1, c2, kernel_size=3, stride=2, activation='leaky',norm_type=norm_type, num_groups=num_groups)
                self.m.append(con3x3)
                self.m.append(Concat())
                c1 = c2*2
                c2 = make_divisible(_c2* 2 * self.width_multiple, 8)
                bottlenect = BottleneckCSP(c1, c2, n=n, shortcut=self.shortcut[idx])
                self.m.append(bottlenect)
                c1 = c2


    def forward(self,x):
        y ,x ,c_idx,outs= x[:2],x[-1],0,[]
        for idx ,m in enumerate(self.m):
            if isinstance(m,Concat):
                x = [x,y[c_idx-1 if c_idx % 2 != 0 else c_idx+1]]
                c_idx+=1
            x = m(x)
            if idx in self.concats:
                y.append(x)
            if isinstance(m,BottleneckCSP):
                outs.append(x)
        if len(outs)==0:
            outs.append(x)

        return outs

    def init_weights(self):
        initialize_weights(self)
