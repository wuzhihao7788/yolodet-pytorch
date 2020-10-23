#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# copy and modify from mmcv.runner.hooks.optinizer.py
# @Time    : 2020/8/10 16:11
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: optinizer_hook.py 
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
from torch.nn.utils import clip_grad
import numpy as np

from .hook import Hook

class OptimizerHook(Hook):

    def __init__(self, grad_clip=None,detect_anomaly=False):
        self.grad_clip = grad_clip
        self.detect_anomaly = detect_anomaly


    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        subdivisions = runner.meta['subdivisions']
        loss = runner.outputs['loss']
        if self.detect_anomaly:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        warmup_max_iters = runner.warmup_max_iters
        accumulate = max(1, np.interp(runner.iter, [0,warmup_max_iters], [1,subdivisions]).round())
        if runner.iter % accumulate == 0:
            if self.detect_anomaly:
                with torch.autograd.detect_anomaly():
                    if self.grad_clip is not None:
                        self.clip_grads(runner.model.parameters())

                    runner.optimizer.step()
                    runner.optimizer.zero_grad()
                    if runner.ema is not None:
                        runner.ema.update(runner.model)
            else:
                # runner.outputs['loss'].backward()
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())

                runner.optimizer.step()
                runner.optimizer.zero_grad()
                if runner.ema is not None:
                    runner.ema.update(runner.model)
