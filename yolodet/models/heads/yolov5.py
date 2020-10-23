#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/23 10:32
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: yolov5.py
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

import torch.nn as nn

import numpy as np
import torch

from yolodet.models.utils.torch_utils import initialize_weights, make_divisible
from yolodet.utils.util import multi_apply
from yolodet.models.heads.base import xyxy2xywh, nms_cpu, soft_nms_cpu
from yolodet.models.heads.yolo import BaseHead


class YOLOv5Head(BaseHead):

    def __init__(self, depth_multiple,width_multiple,**kwargs):
        super(YOLOv5Head, self).__init__(**kwargs)

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        self.m = nn.ModuleList(nn.Conv2d(make_divisible(x * self.width_multiple, 8), len(self.anchor_masks) * (self.base_num + self.num_classes), 1) for x in self.in_channels)  # output conv


    def forward(self, x):
        return [m(x[idx+1]) for idx,m in enumerate(self.m)]

    # def get_eval_bboxes(self,x):
    #     z = []  # inference output
    #     for i, mask in enumerate(self.anchor_masks):
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, len(mask), self.base_num + self.num_classes, ny, nx).permute(0, 1, 3, 4,
    #                                                                                           2).contiguous()
    #         yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    #         grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
    #         y = x[i].sigmoid()
    #         y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_xy) * self.downsample_ratios[i]  # xy 映射回原图位置
    #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.from_numpy(np.array([self.anchors[idx] for idx in mask])).to(
    #             y.device).view(1, -1, 1, 1, 2)  # 映射回原图wh尺寸
    #         z.append(y.view(bs, -1, self.base_num + self.num_classes))
    #
    #     return torch.cat(z, 1), x