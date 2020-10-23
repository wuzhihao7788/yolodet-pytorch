#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/11 9:37
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: class_loss.py 
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

from yolodet.models.loss.base import reduce_loss
from torch import nn
from torch.nn import functional as F

class Class_Loss(nn.Module):

    def __init__(self, reduction='mean',loss_weight=0.5,pos_weight=1.0):
        super(Class_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight=pos_weight
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight]))

    def forward(self,pred,target,num_classes=80):
        device = pred.device
        ft = torch.cuda.FloatTensor if target.is_cuda else torch.Tensor
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([self.pos_weight])).to(device)
        loss = self.loss_weight*(num_classes/80) * BCEcls(pred, target)
        return loss