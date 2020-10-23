#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/27 10:43
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: compose.py 
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

import collections

from yolodet.utils.registry import TRANSFORMS
from yolodet.utils.newInstance_utils import build_from_dict


class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_dict(transform,TRANSFORMS)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
