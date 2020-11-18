#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/21 9:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: YOLOv5Detector
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
from torch import nn
from yolodet.models.detectors.base import BaseDetector


class YOLOv5Detector(BaseDetector):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None):

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        backbone['depth_multiple'] = depth_multiple
        backbone['width_multiple'] = width_multiple

        if neck is not None:
            neck['depth_multiple'] = depth_multiple
            neck['width_multiple'] = width_multiple

        if head is not None:
            head['depth_multiple'] = depth_multiple
            head['width_multiple'] = width_multiple

        super(YOLOv5Detector, self).__init__(backbone=backbone,
                                             neck=neck,
                                             head=head,
                                             pretrained=pretrained)

    def init_weights(self, pretrained=None):

        super(YOLOv5Detector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        if self.with_head:
            if isinstance(self.head, nn.Sequential):
                for m in self.head:
                    m.init_weights()
            else:
                self.head.init_weights()