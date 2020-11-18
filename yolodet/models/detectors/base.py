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
import torch
import torch.nn as nn
from abc import ABCMeta
from yolodet.models.utils.torch_utils import scale_img
from yolodet.utils.registry import BACKBONES, NECKS, HEADS
from yolodet.utils.newInstance_utils import build_from_dict


class BaseDetector(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None):

        super(BaseDetector, self).__init__()
        self.backbone = build_from_dict(backbone, BACKBONES)

        if neck is not None:
            self.neck = build_from_dict(neck, NECKS)

        if head is not None:
            self.head = build_from_dict(head, HEADS)

        self.init_weights(pretrained)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
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

    def forward(self,
                img,
                img_metas,
                return_loss=True,
                **kwargs):

        if kwargs is not None and 'idx' in kwargs:
            idx = kwargs.pop('idx').cpu().numpy()
            img_metas = [img_metas[i] for i in idx]
            for k, v in kwargs.items():
                if isinstance(v, list):
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
                return self.forward_eval(img, img_metas, **kwargs)

            results = self.forward_eval(img, img_metas, **kwargs)[0]
            results = self.forward_test(results, img_metas, **kwargs)

            return results

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_score,
                      gt_class,
                      **kwargs):

        x = self.extract_feat(img)

        losses = dict()

        head_outs = self.head(x)  # y1,y2,y3
        batch_size = len(img)  # input, img_metas, batch_size, gt_bboxes, gt_class, gt_score
        head_loss_inputs = [head_outs, img_metas, batch_size, gt_bboxes, gt_class, gt_score]
        head_loss = self.head.loss(*head_loss_inputs)
        for i in range(len(head_outs)):
            for name, value in head_loss.items():
                losses['Y{}.{}'.format(i + 1, name)] = value[i].detach()
        losses.update(head_loss)

        return losses

    def forward_test(self, x, img_metas, **kwargs):

        head_det_inputs = [x, img_metas, kwargs]
        result = self.head.get_det_bboxes(*head_det_inputs)
        return result

    def forward_eval(self, img, img_metas, **kwargs):

        x = self.extract_feat(img)
        head_outs = self.head(x)  # y1,y2,y3
        return self.head.get_eval_bboxes(head_outs)

    def forward_flops(self, img):

        x = self.extract_feat(img)
        head_outs = self.head(x)  # y1,y2,y3
        return head_outs