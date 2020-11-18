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
