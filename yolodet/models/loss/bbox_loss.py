#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/28 19:18
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: bbox_loss.py
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
import torch

from yolodet.models.loss.base import bbox_iou, reduce_loss
from torch import nn
from torch.nn import functional as F

class IOU_Loss(nn.Module):

    def __init__(self, iou_type='CIOU', reduction='mean', loss_weight=0.05):
        super(IOU_Loss, self).__init__()
        assert iou_type in ('GIOU', 'DIOU', 'CIOU', 'IOU'), 'IOU Loss just support GIOU, DIOU, CIOU, IOU'
        self.giou, self.diou, self.ciou = False, False, False

        if iou_type == 'GIOU':
            self.giou = True
        elif iou_type == 'DIOU':
            self.diou = True
        elif iou_type == 'CIOU':
            self.ciou = True

        self.iou_type = iou_type
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target,weight=None):
        loss,iou = self.iou_loss(pred, target,weight=weight)
        loss = self.loss_weight * loss
        return loss,iou

    def iou_loss(self, pred, target,weight=None):
        iou = bbox_iou(pred.t(), target, x1y1x2y2=False, GIoU=self.giou, DIoU=self.diou, CIoU=self.ciou)
        iou_loss = (1 - iou)
        if weight is not None:
            iou_loss = iou_loss * weight
        # iou_loss = bbox_loss_scale * iou_loss  # 1. target_bbox_confidence作为mask，有物体才计算xxxiou_loss
        loss = reduce_loss(iou_loss, weight=None, reduction=self.reduction)
        return loss,iou # if not torch.isnan(loss) else ft([0]).to(device).mean()

 # 在yolov3中，分类概率和目标物体得分相乘作为最后的置信度，这显然是没有考虑定位的准确度。我们增加了一个额外的IOU预测分支来去衡量检测框定位的准确度，额外引入的参数和FLOPS可以忽略不计
class IOU_Aware_Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(IOU_Aware_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target,ioup):
        loss = self.loss_weight * self.iou_loss(pred, target,ioup)
        return loss

    def iou_loss(self, pred, target,ioup):

        iou = bbox_iou(pred.t(), target, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False).unsqueeze(-1)
        iou = iou.detach()
        iou = torch.clamp(iou,0,1)
        ioup = torch.clamp(ioup,0,1)
        loss_iou_aware = F.binary_cross_entropy_with_logits(input=ioup, target=iou, reduction=self.reduction)

        return loss_iou_aware