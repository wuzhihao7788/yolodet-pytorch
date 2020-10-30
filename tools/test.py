#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/10/28 15:17
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: test.py
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from yolodet.models.utils.torch_utils import select_device

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
# # config = '/disk2/project/pytorch-YOLOv4/cfg/yolov4_hand_gpu.py'
# config = '/disk2/project/pytorch-YOLOv4/cfg/yolov5_hand_gpu.py'
# config = '/disk2/project/pytorch-YOLOv4/cfg/ppyolo_hand_gpu.py'
# config = '/disk2/project/mmdetection/mount/pytorch-YOLOv4/cfg/yolov5_coco_gpu.py'
# # checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov4-hand/latest.pth'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/ppyolo_hand/latest.pth'
# # checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/test/latest.pth'
# config = '/disk2/project/mmdetection/mount/pytorch-YOLOv4/cfg/yolov5_coco_gpu.py'
# checkpoint = '/disk2/project/mmdetection/mount/pytorch-YOLOv4/work_dirs/coco/epoch_10.pth'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/epoch_20.pth'
# cfg = Config.fromfile(config)
# cfg.data.val.train = False
# val_dataset = build_from_dict(cfg.data.val, DATASET)
# val_dataloader = build_dataloader(val_dataset, data=cfg.data,shuffle=False)
#
# model = init_detector(config, checkpoint=checkpoint, device='cuda:0')
#
# results = single_gpu_test(model, val_dataloader,half=True, save=False)
#
# print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--config', type=str, default='yolov5_coco.py', help='model config file')
    parser.add_argument('--checkpoint',type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--coco-val-path', default='/disk2/project/coco/annotations/', help='cocoapi val JSON file Path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--half', action='store_true', help='fp16 half precision')
    opt = parser.parse_args()

    print(opt)

    cfg = Config.fromfile(opt.config)
    cfg.data.val.train = False
    val_dataset = build_from_dict(cfg.data.val, DATASET)
    val_dataloader = build_dataloader(val_dataset, data=cfg.data, shuffle=False)
    device = select_device(opt.device)
    # model = init_detector(opt.config, checkpoint=opt.checkpoint, device=device)
    model = init_detector(opt.config, checkpoint=opt.checkpoint, device=device)
    result = single_gpu_test(model, val_dataloader, half=opt.half,conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=opt.merge,
                     save_json=opt.save_json, augment=opt.augment, verbose=opt.verbose,coco_val_path=opt.coco_val_path)

    print(result)
