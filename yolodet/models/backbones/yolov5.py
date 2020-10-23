#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/22 10:10
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

import torch
from torch import nn

from yolodet.models.backbones.base import BottleneckCSP, Focus, DarknetConv2D_Norm_Activation
from yolodet.models.utils.torch_utils import make_divisible, initialize_weights


class YOLOv5Darknet(nn.Module):
    def __init__(self,depth_multiple,width_multiple,focus,in_channels=3,bottle_depths=[3, 9, 9, 3],out_channels = [128,256,512,1024],spp=[5, 9, 13],shortcut = [True,True,True,False], out_indices=(2, 3, 4,),norm_type='BN',num_groups=None):
        super(YOLOv5Darknet,self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.bottle_depths = bottle_depths
        self.out_channels = out_channels
        assert len(self.bottle_depths) == len(self.out_channels),'len(self.bottle_depths) must = len(self.bottle_out_channels)'
        self.out_indices = out_indices

        assert norm_type in ('BN','GN'),'norm_type must BN or GN'
        # self.norm_type = norm_type
        # self.num_groups = num_groups
        _in_channels = in_channels
        if focus is not None:
            c2 = focus[0]
            c2 = make_divisible(c2 * self.width_multiple, 8)
            self.focus = Focus(in_channels,c2,*focus[1:],activation='leaky',norm_type=norm_type,num_groups=num_groups)
            _in_channels = c2

        self.out_indices = []
        self.m = nn.ModuleList()
        for i ,depth in enumerate(self.bottle_depths):
            c2 = self.out_channels[i]
            c2 = make_divisible(c2 * self.width_multiple, 8)
            n = max(round(depth * self.depth_multiple), 1) if depth > 1 else depth  # depth gain
            con3x3 = DarknetConv2D_Norm_Activation(_in_channels, c2, kernel_size=3, stride=2, activation='leaky',
                                                        norm_type=norm_type, num_groups=num_groups)
            self.m.append(con3x3)
            _in_channels = c2
            if i == len(self.bottle_depths)-1 and spp is not None:
                spp = SPP(_in_channels, c2, k=spp)
                self.m.append(spp)
                _in_channels = c2

            bottlenect = BottleneckCSP(_in_channels, c2, n=n,shortcut=shortcut[i])
            _in_channels = c2
            self.m.append(bottlenect)

            if i+1 in out_indices:
                self.out_indices.append(len(self.m)-1)

        self.init_weights()


    def forward(self, x):
        out = self.focus(x)
        outs = []
        for idx, _m in enumerate(self.m):
            out = _m(out)
            if idx in self.out_indices:
                outs.append(out)

        if len(outs)==0:
            outs.append(out)
        return outs

    def init_weights(self,pretrained=None):
        initialize_weights(self)

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = DarknetConv2D_Norm_Activation(c1, c_, 1, 1, activation='leaky')
        self.cv2 = DarknetConv2D_Norm_Activation(c_ * (len(k) + 1), c2, 1, 1,activation='leaky')
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))