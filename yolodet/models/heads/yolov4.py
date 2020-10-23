#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: yolov4.py
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
from yolodet.models.heads.yolo import BaseHead
from yolodet.models.heads.base import Y
class YOLOv4Head(BaseHead):

    def __init__(self,**kwargs):
        super(YOLOv4Head, self).__init__(**kwargs)
        self.y1 = Y(self.in_channels[-1], self.out_channels[0],norm_type=self.norm_type,num_groups=self.num_groups)
        self.y2 = Y(self.in_channels[-2], self.out_channels[1],norm_type=self.norm_type,num_groups=self.num_groups)
        self.y3 = Y(self.in_channels[-3], self.out_channels[2],norm_type=self.norm_type,num_groups=self.num_groups)

    def forward(self, x):
        # x.size = 3 The last three predict layers
        y1 = self.y1(x[0])  # 19x19xout_channels if img size is 608
        y2 = self.y2(x[1])  # 38x38xout_channels if img size is 608
        y3 = self.y3(x[2])  # 72x72xout_channels if img size is 608
        return [y1, y2, y3]