#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: base.py
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

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDetector(nn.Module,metaclass=ABCMeta):

    def __init__(self):
        super(BaseDetector,self).__init__()


    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            if img_metas is None or kwargs is None or not len(kwargs):
                return self.forward_flops(img)
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    @abstractmethod
    def forward_train(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward_flops(self, img):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
