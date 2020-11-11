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

import torch
import torch.nn as nn

from yolodet.models.utils.torch_utils import scale_img


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
        if kwargs is not None and 'idx' in kwargs:
            idx = kwargs.pop('idx').cpu().numpy()
            img_metas = [img_metas[i] for i in idx]
            for k,v in kwargs.items():
                if isinstance(v,list):
                    kwargs[k] = [v[i] for i in idx]
        if return_loss:
            if img_metas is None or kwargs is None or not len(kwargs):
                return self.forward_flops(img)
            return self.forward_train(img, img_metas, **kwargs)
        else:
            if 'augment' in kwargs and kwargs['augment']:
                img_size = img.shape[-2:]  # height, width
                s = [0.83, 0.67]  # scales
                y = []
                for i, xi in enumerate((img,
                                        scale_img(img.flip(3), s[0]),  # flip-lr and scale
                                        scale_img(img, s[1]),  # scale
                                        )):
                    y.append(self.forward_eval(xi, img_metas, **kwargs)[0])

                y[1][..., :4] /= s[0]  # scale
                y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
                y[2][..., :4] /= s[1]  # scale
                if 'eval' in kwargs and kwargs['eval']:
                    return torch.cat(y, 1), None  # augmented inference, train
                else:
                    return self.forward_test(torch.cat(y, 1), img_metas, **kwargs)
            elif 'eval' in kwargs and kwargs['eval']:
                return self.forward_eval(img,img_metas,**kwargs)

            return self.forward_test(self.forward_eval(img,img_metas,**kwargs)[0], img_metas, **kwargs)

    @abstractmethod
    def forward_train(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward_eval(self, img, img_metas, **kwargs):
        pass


    @abstractmethod
    def forward_flops(self, img):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
