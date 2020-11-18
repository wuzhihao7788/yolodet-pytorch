#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/31 13:38
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: PPYOLODetector
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


class PPYOLODetector(BaseDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None):

        super(PPYOLODetector, self).__init__(backbone=backbone,
                                             neck=neck,
                                             head=head,
                                             pretrained=pretrained)

    def init_weights(self, pretrained=None):

        super(PPYOLODetector, self).init_weights(pretrained)
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