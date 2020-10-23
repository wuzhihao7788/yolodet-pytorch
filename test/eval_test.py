#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/21 16:40
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: eval_test.py 
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
from yolodet.apis.inference import init_detector
from yolodet.apis.test import single_gpu_test
from yolodet.dataset.loader.build_dataloader import build_dataloader
from yolodet.utils.config import Config
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.utils.registry import DATASET
# config = '/disk2/project/pytorch-YOLOv4/cfg/yolov4_hand_gpu.py'
config = '/disk2/project/pytorch-YOLOv4/cfg/yolov5_hand_gpu.py'
config = '/disk2/project/pytorch-YOLOv4/cfg/ppyolo_hand_gpu.py'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov4-hand/latest.pth'
checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/ppyolo_hand/latest.pth'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/test/latest.pth'
cfg = Config.fromfile(config)
cfg.data.val.train = False
val_dataset = build_from_dict(cfg.data.val, DATASET)
val_dataloader = build_dataloader(val_dataset, data=cfg.data,shuffle=False)

model = init_detector(config, checkpoint=checkpoint, device='cuda:0')

results = single_gpu_test(model, val_dataloader, show=False)
print(results)

