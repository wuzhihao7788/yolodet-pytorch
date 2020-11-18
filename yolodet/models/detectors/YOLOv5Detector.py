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
from yolodet.models.detectors.base import BaseDetector


class YOLOv5Detector(BaseDetector):

    def __init__(self,
                 depth_multiple,
                 width_multiple,
                 backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None):

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