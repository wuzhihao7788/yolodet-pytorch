#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/10 19:22
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: ppyolo.py
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
import torch.nn as nn

from yolodet.models.heads.yolo import BaseHead
from yolodet.models.backbones.base import DarknetConv2D_Norm_Activation
from yolodet.models.necks.base import CoordConv
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.utils.registry import LOSS
from yolodet.utils.util import multi_apply
import numpy as np



class Y(nn.Module):
    def __init__(self, in_channels, out_channels,activation='leaky',norm_type='BN',num_groups=None,coord_conv=True):
        super(Y, self).__init__()
        self.c_conv3x3 = CoordConv(in_channels=in_channels, out_channels=in_channels*2,kernel_size=3,activation=activation,norm_type=norm_type,num_groups=num_groups,coord_conv=coord_conv)
        self.liner = DarknetConv2D_Norm_Activation(in_channels * 2, out_channels, 1, 1, activation='linear', norm_type=None,bias=True)

    def forward(self, x):
        x = self.c_conv3x3(x)
        x = self.liner(x)
        return x


class PPHead(BaseHead):

    def __init__(self,aware_Loss = dict(type='IOU_Aware_Loss'),scale_x_y=1.05,iou_aware=True,iou_aware_factor=0.4,coord_conv=True,xywh_loss=True,**kwargs):
        super(PPHead, self).__init__(**kwargs)

        self.coord_conv = coord_conv
        self.scale_x_y = scale_x_y
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.xywh_loss = xywh_loss

        out_channels = []
        self.base_num = 5
        if iou_aware:
            self.base_num = 6
        for mask in self.anchor_masks:
            out_channels.append(len(mask) * (self.base_num + self.num_classes))

        self.y1 = Y(self.in_channels[-1], out_channels[0],norm_type=self.norm_type,num_groups=self.num_groups,coord_conv=self.coord_conv)
        self.y2 = Y(self.in_channels[-2], out_channels[1],norm_type=self.norm_type,num_groups=self.num_groups,coord_conv=self.coord_conv)
        self.y3 = Y(self.in_channels[-3], out_channels[2],norm_type=self.norm_type,num_groups=self.num_groups,coord_conv=self.coord_conv)

        if aware_Loss is not None:
            self.aware_Loss = build_from_dict(aware_Loss, LOSS)

    def forward(self, x):
        # x.size = 3 The last three predict layers
        y1 = self.y1(x[0])  # 19x19xout_channels if img size is 608
        y2 = self.y2(x[1])  # 38x38xout_channels if img size is 608
        y3 = self.y3(x[2])  # 72x72xout_channels if img size is 608
        return [y1, y2, y3]

    def loss(self, input, img_metas, batch_size, gt_bboxes, gt_class, gt_score):
        pred = self.get_pred(input)
        indices,tbox,tcls,anchors,ignore_mask = self.get_target(pred = pred,img_metas=img_metas, batch_size=batch_size, gt_bbox=gt_bboxes, gt_class=gt_class, gt_score=gt_score)
        if self.yolo_loss_type is not None:
            bbox_loss, confidence_loss, class_loss = multi_apply(self.loss_single, pred, indices,tbox,tcls,anchors,self.conf_balances,ignore_mask)
            return dict(bbox_loss=bbox_loss, confidence_loss=confidence_loss, class_loss=class_loss)
        else:
            bbox_loss, confidence_loss, class_loss,aware_loss,xy_loss,wh_loss = multi_apply(self.loss_single, pred, indices,tbox,tcls,anchors,self.conf_balances,ignore_mask)
            return dict(bbox_loss=bbox_loss, confidence_loss=confidence_loss, class_loss=class_loss,aware_loss=aware_loss,xy_loss=xy_loss,wh_loss=wh_loss)

    def loss_single(self, pred, indices,tbox,tcls,anchors,conf_balances,ignore_mask):
        if self.yolo_loss_type in ['yolov4','yolov5']:
            return super().loss_single(pred, indices,tbox,tcls,anchors,conf_balances,ignore_mask)

        device = pred.device
        b, a, gj, gi = indices
        tobj = torch.zeros_like(pred[..., 4]).to(device)  # target obj
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj,laware,lxy,lwh = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        gird_h,gird_w = pred.shape[-3:-1]

        nb = b.shape[0]  # number of targets
        if nb:
            ps = pred[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1

            tscale_tobj = torch.ones_like(ps[..., 4]).to(device)  # target obj
            if self.bbox_weight:
                tscale_tobj = (2.0 - (1.0 * tbox[..., 2:3] * tbox[..., 3:4]) / (gird_h * gird_w)).reshape(-1)

            pxy = torch.sigmoid(ps[..., :2])
            if self.scale_x_y > 1:
                pxy = self.scale_x_y * pxy - 0.5 * (self.scale_x_y -1.0)

            pwh = torch.exp(ps[..., 2:4]) * torch.from_numpy(anchors).to(device)
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            lbox += self.bbox_loss(pbox,tbox,tscale_tobj)[0]

            t = torch.full_like(ps[:, self.base_num:], 0).to(device)  # targets
            t[range(nb), tcls.squeeze()] = 1
            if self.label_smooth:  # https://arxiv.org/pdf/1812.01187.pdf
                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                smooth_onehot = t * (1 - self.deta) + self.deta * torch.from_numpy(uniform_distribution).to(device)
                t = smooth_onehot
            lcls += self.class_loss(ps[:, self.base_num:], t, self.num_classes)
            if self.xywh_loss:
                gwh = torch.log(tbox[...,2:4]/torch.from_numpy(anchors).to(device))
                loss_x = torch.abs(pxy[...,0] - tbox[...,0])*tscale_tobj
                loss_y = torch.abs(pxy[...,1] - tbox[...,1])*tscale_tobj

                loss_w = torch.abs(ps[..., 2] - gwh[...,0])*tscale_tobj
                loss_h = torch.abs(ps[..., 3] - gwh[...,1])*tscale_tobj

                lxy += (loss_x+loss_y).mean()
                lwh += (loss_w+loss_h).mean()

            if self.iou_aware:
                ioup = torch.sigmoid(ps[..., 5:self.base_num])
                laware += self.aware_Loss(pred=pbox, target=tbox, ioup=ioup)

        lobj += self.confidence_loss(pred[..., 4], tobj, ignore_mask) * conf_balances

        return lbox, lobj, lcls,laware,lxy,lwh

    def get_eval_bboxes(self,x):

        if self.yolo_loss_type in ['yolov4','yolov5']:
            return super().get_eval_bboxes(x)

        z = []  # inference output
        for i, mask in enumerate(self.anchor_masks):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, len(mask), self.base_num + self.num_classes, ny, nx).permute(0, 1, 3, 4,
                                                                                              2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i]
            pxy = torch.sigmoid(y[..., :2])
            if self.scale_x_y > 1:
                pxy = self.scale_x_y * pxy - 0.5 * (self.scale_x_y -1.0)
            y[..., :2] = (pxy + grid_xy) * self.downsample_ratios[i]  # 映射回原图位置
            y[..., 2:4] = torch.exp(y[..., 2:4]) * torch.from_numpy(np.array([self.anchors[idx] for idx in mask])).to(
                y.device).view(1, -1, 1, 1, 2)  # 映射回原图wh尺寸
            y[..., 4:] = torch.sigmoid(y[..., 4:])
            if self.iou_aware and self.base_num == 6 and self.iou_aware_factor<1:
                obj = y[..., 4:5]
                ioua = y[..., 5:self.base_num]
                new_obj = torch.pow(obj, (1 - self.iou_aware_factor)) * torch.pow(ioua, self.iou_aware_factor)
                eps = 1e-7
                new_obj = torch.clamp(new_obj, eps, 1 / eps)
                one = torch.ones_like(new_obj)
                new_obj = torch.clamp((one / new_obj - 1.0), eps, 1 / eps)
                new_obj = -torch.log(new_obj)
                y[..., 4:5] = new_obj
                y = torch.cat([y[..., :5],y[..., self.base_num:]],dim=-1)
                z.append(y.view(bs, -1, self.base_num-1 + self.num_classes))
            else:
                z.append(y.view(bs, -1, self.base_num + self.num_classes))

        return torch.cat(z, 1), x
