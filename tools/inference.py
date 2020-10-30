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
import argparse
import random
import time

from yolodet.apis.inference import inference_detector, init_detector, show_result
import os
import numpy as np

from yolodet.models.utils import torch_utils
from yolodet.models.utils.torch_utils import select_device


def get_file_realpath(src, *tar):
    for root, _, files in os.walk(src):
        for fn in files:
            fn_name, fn_ext = os.path.splitext(fn)
            if fn_ext.lower() not in tar:
                continue
            yield os.path.join(root, fn)


def inference(config,checkpoint,img_path,device='cuda:0',half=False,augment=False,scores_thr=0.3,merge=False,save_json=False,save_file=False,save_path='',show=False):

    device = select_device(device)
    model = init_detector(config, checkpoint=checkpoint, device=device)
    t0 = time.time()

    if os.path.isdir(img_path):
        imgs_paths = get_file_realpath(img_path, *[".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"])
        for i, img in enumerate(imgs_paths):
            basename = os.path.basename(img)
            result,t = inference_detector(model, img=img, scores_thr=scores_thr,augment=augment, half=half,merge=merge)

            #print(result)
            # Print time (inference + NMS)
            print('%s Done , object num : %s .time:(%.3fs)' % (basename,len(result), t))
            show_result(img, result, model.CLASSES, show=show,save_json=save_json,save_file=save_file,out_path=save_path,file_name=basename)

    else:
        basename = os.path.basename(img_path)
        t1 = torch_utils.time_synchronized()
        result = inference_detector(model, img=img_path, scores_thr=scores_thr, augment=augment, half=half, merge=merge)
        t2 = torch_utils.time_synchronized()
        # print(result)
        # Print time (inference + NMS)
        print('%s Done , object num : %s .time:(%.3fs)' % (basename, len(result), t2 - t1))
        show_result(img_path, result, model.CLASSES, show=show, save_json=save_json, save_file=save_file,
                    out_path=save_path, file_name=basename)


    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='inference.py')
    parser.add_argument('--config', type=str, default='yolov5_coco.py', help='model config file')
    parser.add_argument('--checkpoint',type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--img-path',type=str, default='inference/images', help='input image path or image file')
    parser.add_argument('--scores-thr', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--save-json', action='store_true', help='save detect result by json')
    parser.add_argument('--save-file', action='store_true', help='save detect result to image')
    parser.add_argument('--save-path', default='inference/result', help='save detect result file path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--half', action='store_true', help='fp16 half precision')
    parser.add_argument('--show', action='store_true', help='show detect result in image by opencv')
    opt = parser.parse_args()

    print(opt)

    inference(opt.config, opt.checkpoint, opt.img_path, device=opt.device, half=opt.half, augment=opt.augment, scores_thr=opt.scores_thr, merge=opt.merge,
              save_json=opt.save_json, save_file=opt.save_file, save_path=opt.save_path, show=opt.show)