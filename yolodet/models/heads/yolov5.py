#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/23 10:32
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: yolov5.py
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
from yolodet.models.heads.yolo import BaseHead
from yolodet.models.utils.torch_utils import make_divisible


class YOLOv5Head(BaseHead):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 **kwargs):

        super(YOLOv5Head, self).__init__(**kwargs)

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        # output conv
        # self.m = nn.ModuleList(nn.Conv2d(make_divisible(x * self.width_multiple, 8),
        # len(self.anchor_masks) * (self.base_num + self.num_classes), 1) for x in self.in_channels)
        self.model_list = nn.ModuleList()
        output = len(self.anchor_masks) * (self.base_num + self.num_classes)
        for idx, in_channels in enumerate(self.in_channels):
            divisible = make_divisible(in_channels * self.width_multiple, 8)
            self.model_list.append(nn.Conv2d(divisible, output, 1))

    def forward(self, x):
        return [m(x[idx+1]) for idx, m in enumerate(self.model_list)]