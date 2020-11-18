#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/28 18:24
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: YOLOv4Detector.py
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


class YOLOv4Detector(BaseDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None):

        super(YOLOv4Detector, self).__init__(backbone=backbone,
                                             neck=neck,
                                             head=head,
                                             pretrained=pretrained)