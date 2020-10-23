#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/28 16:42
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: dataset_test.py
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
from yolodet.utils.newInstance_utils import build_from_dict

from yolodet.utils.registry import DATASET
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def draw_box(img, bboxes,class_name,gt_class,gt_score):
    # for b in bboxes:
    #     img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    for idx,ind in enumerate(gt_class):
        if ind>-1:
            b = bboxes[idx]
            img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img,class_name[int(ind)]+'|'+str(gt_score[idx]),(b[0], b[1]), font, 0.3, (0, 0, 255), 1)
    return img

dataset = build_from_dict(dataset_test.data['train'],DATASET)


for i in range(30):
    result = dataset.__getitem__(i)
    img = result['img']
    gt_bboxes = result['gt_bboxes']
    gt_score = result['gt_score']
    gt_bboxes[:, [1, 3]] *= img.shape[0]  # height
    gt_bboxes[:, [0, 2]] *= img.shape[1]  # width
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    a = draw_box(img.copy(), gt_bboxes.astype(np.int32),dataset.CLASSES,result['gt_class'],gt_score)
    for b in gt_bboxes.astype(np.int32):
        if b[0] !=0 or  b[1] !=0  or  b[2] !=0  or  b[3] !=0:
            print(b)
    cv2.imwrite(os.path.join('/disk2/project/test/v2.0/yolov5/dataset/123/test',str(i)+'.jpg'),a)
    print('-------------------------------------------')