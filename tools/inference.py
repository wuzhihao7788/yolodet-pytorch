#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/13 15:09
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: inference.py
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
import random

from yolodet.apis.inference import inference_detector, init_detector, show_result
import os
import numpy as np


def get_file_realpath(src, *tar):
    for root, _, files in os.walk(src):
        for fn in files:
            fn_name, fn_ext = os.path.splitext(fn)
            if fn_ext.lower() not in tar:
                continue
            # print(os.path.join(root, fn))
            yield os.path.join(root, fn)


config = '/disk2/project/pytorch-YOLOv4/cfg/yolov4_hand_gpu.py'
config = '/disk2/project/pytorch-YOLOv4/cfg/ppyolo_hand_gpu.py'
# config = '/disk2/project/mmdetection/mount/pytorch-YOLOv4/cfg/yolov4_coco_gpu.py'
# checkpoint = '/disk2/project/mmdetection/mount/pytorch-YOLOv4/work_dirs/coco/epoch_9.pth'
checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov4_hand_lossv5/latest.pth'
checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/ppyolo_hand_v5loss/latest.pth'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov4-hand/latest.pth'
img = '/disk2/project/pytorch-YOLOv4/data2/imgs/00005.jpg'
# image_path = '/disk2/project/pytorch-YOLOv4/data_test'
# image_path = '/disk2/project/coco/val2017'
image_path = '/disk2/project/mmdetection/mount/hand/ttt'

imgs_paths = get_file_realpath(image_path, *[".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"])
model = init_detector(config, checkpoint=checkpoint, device='cuda:0')
imgs_paths = list(imgs_paths)
random.shuffle(imgs_paths)
# for i ,img in enumerate(imgs_paths):

for i, img in enumerate(imgs_paths):
    print(img)
    basename = os.path.basename(img)
    # conf_thres, iou_thres, merge = 0.001, 0.6, False
    result = inference_detector(model, img=img, scores_thr=0.3, half=False)
    print(result)
    show_result(img, result, model.CLASSES, show=False,
                out_file=os.path.join('/disk2/project/pytorch-YOLOv4/result', basename))