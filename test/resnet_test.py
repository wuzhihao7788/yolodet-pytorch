#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/31 17:16
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: resnet_test
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
#-*- coding:utf-8 -*-
import random
imgsz = 640
imgs = []
sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
sf = sz / max(imgs.shape[2:])  # scale factor
if sf != 1:
    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)