#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/11 9:40
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: confidence_loss.py 
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
from yolodet.models.loss.base import reduce_loss
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class Conf_Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0,pos_weight=1.0):
        super(Conf_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight

    def forward(self, pred, target, ignore_mask=None):
        device = pred.device
        if ignore_mask is not None:
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight]),reduction='none').to(device)
            obj_loss = BCEobj(pred,target)
            pos_loss = target*obj_loss
            neg_loss = (1-target)*ignore_mask.squeeze(dim=-1)*obj_loss
            confidence_loss = (pos_loss+neg_loss).mean()*self.loss_weight
        else:
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight])).to(device)
            confidence_loss = BCEobj(pred, target) * self.loss_weight
        return confidence_loss
