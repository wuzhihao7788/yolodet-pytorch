#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/13 14:55
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
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from yolodet.utils.config import Config
from yolodet.utils.registry import DETECTORS
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.apis.checkpoint import load_checkpoint
from yolodet.dataset.pipelines.compose import Compose
import cv2
from tools.file_utils import mkdir_or_exist
import os.path as osp
from yolodet.models.utils import torch_utils

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detectors from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detectors.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = attempt_load(checkpoint,map_location=device)
    model.cfg = config  # save the config in the model for convenience
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = cv2.imdecode(np.fromfile(results['img'], dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img,scores_thr=0.3,augment=False, half=False,merge=False):
    """Inference image(s) with the detectors.

    Args:
        model (nn.Module): The loaded detectors.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    img = data['img']
    half = device.type != 'cpu' and half  # half precision only supported on CUDA, CUDA >=10.1,pytorch >=1.5
    # ori_img = data.pop('ori_img')
    if len(img.shape) == 3:  # cv2 image
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        elif isinstance(img,torch.Tensor):
            img = img.unsqueeze(0)
    elif len(img.shape) == 4:
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).half() if half else img.to(device).float()  # uint8 to fp16/32
    data['img'] = img
    assert isinstance(scores_thr,float),'scores_thr must float'
    if scores_thr>1 or scores_thr<0:
        raise Exception('scores_thr must >=0 or <=1')

    data['scores_thr'] = scores_thr
    if augment:
        data['augment'] = augment
    if merge:
        data['merge'] = merge
    if half:
        model.half()  # to FP16
    # forward the model
    with torch.no_grad():
        t1 = torch_utils.time_synchronized()
        result = model(return_loss=False, rescale=True, **data)
        t2 = torch_utils.time_synchronized()
    return result ,t2 - t1


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = nn.ModuleList()
    for w in weights if isinstance(weights, list) else [weights]:

        model.append(torch.load(w, map_location=map_location)['model'].float().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


# TODO: merge this method with the one in BaseDetector
def show_result(img,result,class_names,show=True,save_json=False,save_file=False,out_path=None,file_name=None):
    # assert isinstance(class_names, (None,tuple, list))
    assert isinstance(img,(str,np.ndarray))
    if isinstance(img,str):
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
    img = img.copy()

    if save_json and out_path and file_name:
        mkdir_or_exist(out_path)
        basename = os.path.basename(file_name)
        bname = os.path.splitext(basename)[0]
        with open(os.path.join(out_path,bname+'.json'),'w',encoding='utf8') as f:
            json.dump(result, f)

    for rslt in result:
        label = rslt['label']
        score = rslt['score']
        bbox_int = rslt['bbox']
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, color=(0,0,255), thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        label_text += '|{:.02f}'.format(score)
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,255,0))

    if save_file and out_path and file_name:
        mkdir_or_exist(out_path)
        cv2.imwrite(os.path.join(out_path,file_name),img)
    if show:
        win_name = 'inference result images'
        wait_time = 0
        cv2.imshow(win_name, img)
        if wait_time == 0:  # prevent from hangning if windows was closed
            while True:
                ret = cv2.waitKey(1)

                closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                # if user closed window or if some key pressed
                if closed or ret != -1:
                    break
        else:
            ret = cv2.waitKey(wait_time)

    return img

# def show_result_pyplot(img,result,class_names,score_thr=0.01,fig_size=(15, 10)):
#     img = show_result(img, result, class_names, score_thr=score_thr, show=False)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=fig_size)
#     plt.imshow(img)
#     plt.show()
