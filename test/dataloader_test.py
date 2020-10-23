#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/9/28 12:05
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: dataloader_test.py
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

from cfg import dataset_test
from yolodet.dataset.loader.build_dataloader import build_dataloader
from yolodet.models.heads.base import xyxy2xywh
from yolodet.utils.config import Config
from yolodet.utils.newInstance_utils import build_from_dict

from yolodet.utils.registry import DATASET
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def draw_box(img, bboxes,wh):
    h,w = wh
    # for b in bboxes:
    #     img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    for idx,bbox in enumerate(bboxes):
        # x1, y1, x2, y2 = bbox
        cx, cy, _w, _h = bbox
        x, y = cx - _w / 2, cy - _h / 2
        # x1,y1,x2,y2 = int(x1*w),int(y1*h),int(x2*w),int(y2*h)
        x1,y1,x2,y2 = x,y, x+_w,y+_h
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img

file = '/disk2/project/pytorch-YOLOv4/cfg/dataset_test.py'

cfg = Config.fromfile(file)

dataset = build_from_dict(cfg.data.train,DATASET)

dataloader = build_dataloader(dataset,data=cfg.data)

for i, data_batch in enumerate(dataloader):
    if i>30:
        break
    for idx,data in enumerate(data_batch['img']):
        gt = data_batch['gt_bboxes'][idx]
        gt_xywh = xyxy2xywh(gt)  # x,y ,w, h
        n_gt = (gt.sum(dim=-1) > 0).sum(dim=-1)
        n = int(n_gt)
        if n == 0:
            continue
        gt = gt[:n].cpu().numpy()
        gt_xywh = gt_xywh[:n].cpu().numpy()
        data = data.cpu().numpy()*255
        data = data.transpose(1, 2, 0)
        h,w = data.shape[:2]
        a = draw_box(data.copy(), gt_xywh,(h,w))
        cv2.imwrite(os.path.join('/disk2/project/test/v2.0/yolov5/dataset/123/test', str(i)+'+'+str(idx) + '.jpg'), a)
# for i in range(30):
#     result = dataset.__getitem__(i)
#     img = result['img']
#     gt_bboxes = result['gt_bboxes']
#     gt_score = result['gt_score']
#     gt_bboxes[:, [1, 3]] *= img.shape[0]  # height
#     gt_bboxes[:, [0, 2]] *= img.shape[1]  # width
#     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
#     a = draw_box(img.copy(), gt_bboxes.astype(np.int32),dataset.CLASSES,result['gt_class'],gt_score)
#     for b in gt_bboxes.astype(np.int32):
#         if b[0] !=0 or  b[1] !=0  or  b[2] !=0  or  b[3] !=0:
#             print(b)
#     cv2.imwrite(os.path.join('/disk2/project/test/v2.0/yolov5/dataset/123/test',str(i)+'.jpg'),a)
#     print('-------------------------------------------')