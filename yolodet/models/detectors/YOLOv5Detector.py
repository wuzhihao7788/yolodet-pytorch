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

import time

from torch import nn

from yolodet.models.detectors.base import BaseDetector
from yolodet.utils.registry import BACKBONES, NECKS, HEADS
from yolodet.utils.newInstance_utils import build_from_dict


class YOLOv5Detector(BaseDetector):

    def __init__(self,depth_multiple,width_multiple,backbone=None,neck=None,head=None,pretrained=None):
        super(YOLOv5Detector, self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        backbone['depth_multiple'] = depth_multiple
        backbone['width_multiple'] = width_multiple
        self.backbone = build_from_dict(backbone,BACKBONES)
        if neck is not None:
            neck['depth_multiple'] = depth_multiple
            neck['width_multiple'] = width_multiple
            self.neck = build_from_dict(neck,NECKS)
        if head is not None:
            head['depth_multiple'] = depth_multiple
            head['width_multiple'] = width_multiple
            self.head = build_from_dict(head,HEADS)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        super(YOLOv5Detector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        if self.with_head:
            if isinstance(self.head, nn.Sequential):
                for m in self.head:
                    m.init_weights()
            else:
                self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+necks
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,img, img_metas,gt_bboxes,gt_score,gt_class, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        head_outs = self.head(x)# y1,y2,y3
        batch_size = len(img)#input, img_metas, batch_size, gt_bboxes, gt_class, gt_score
        head_loss_inputs = [head_outs,img_metas, batch_size, gt_bboxes, gt_class, gt_score]
        head_loss = self.head.loss(*head_loss_inputs)
        for i in range(len(head_outs)):
            for name, value in head_loss.items():
                losses['Y{}.{}'.format(i+1, name)] = value[i].detach()
        losses.update(head_loss)
        return losses


    def forward_test(self, x, img_metas, **kwargs):
        head_det_inputs = [x, img_metas,kwargs]
        result = self.head.get_det_bboxes(*head_det_inputs)
        return result

    def forward_flops(self, img):
        x = self.extract_feat(img)
        head_outs = self.head(x)# y1,y2,y3
        return head_outs

    def forward_eval(self, img, img_metas, **kwargs):
        x = self.extract_feat(img)
        head_outs = self.head(x)# y1,y2,y3
        return self.head.get_eval_bboxes(head_outs)