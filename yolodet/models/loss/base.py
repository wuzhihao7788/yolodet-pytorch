#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/29 15:18
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
import math

import torch
import numpy as np



def bbox_ciou(boxes1, boxes2,GIoU=False, DIoU=False, CIoU=True):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = torch.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = torch.cat((torch.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             torch.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((torch.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             torch.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = torch.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = torch.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = right_down - left_up
    inter_section = torch.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    if GIoU or DIoU or CIoU:
        # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
        enclose_top_left = torch.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        enclose_bottom_right = torch.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])
        if GIoU:
            area_c = torch.prod(enclose_bottom_right - enclose_top_left, 2)  # convex area
            return iou - (area_c - union_area) / area_c  # GIoU
        # 包围矩形的对角线的平方
        enclose_wh = enclose_bottom_right - enclose_top_left
        # enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)

        # 两矩形中心点距离的平方
        p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)
        # c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)+ 1e-16
            if DIoU:

                return iou - p2 / enclose_c2  # DIoU

            if CIoU:
            # 增加av。加上除0保护防止nan。
                atan1 = torch.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
                atan2 = torch.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
                v = 4.0 * torch.pow(atan1 - atan2, 2) / (np.math.pi ** 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * alpha * v

                return ciou
    return iou

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou
def box_iou(box1, box2,xyxy=True):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        if xyxy:
        # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])
        else:
            x1, x2 = box[0] - box[2] / 2, box[0] + box[2] / 2
            y1, y2 = box[1] - box[3] / 2, box[1] + box[3] / 2
            return (x2-x1)*(y2-y1)

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    if xyxy:
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    else:
        # Intersection area
        tl = torch.max((box1[:, None, :2] - box1[:, None, 2:] / 2),(box2[:, :2] - box2[:, 2:] / 2))#x1,y1
        # # intersection bottom right
        br = torch.min((box1[:, None, :2] + box1[:, None, 2:] / 2),(box2[:, :2] + box2[:, 2:] / 2))#x2,y2
        inter = (br - tl).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def reduce_loss(loss, weight=None, reduction='mean'):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    if weight:
        loss = loss * weight
    reduction_enum = get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret