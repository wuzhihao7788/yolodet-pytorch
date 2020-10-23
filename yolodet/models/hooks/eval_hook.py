#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 18:55
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: eval_hook.py 
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

import os.path as osp

import numpy as np

from .hook import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):

    def __init__(self, dataloader, interval=1,save_best=True, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_best = save_best
        self.best = None

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from yolodet.apis.test import single_gpu_test
        if runner.ema is not None:
            model = runner.ema.ema.module if hasattr(runner.ema, 'module') else runner.ema.ema
            results = single_gpu_test(model, self.dataloader)
        else:
            results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        results, maps, times = results
        if self.save_best:
            def fitness(x):
                # Returns fitness (for use with results.txt or evolve.txt)
                w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
                return (x[:, :4] * w).sum(1)

            # Update best mAP
            _best = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if self.best is None or self.best<_best:
                self.best = _best

            if self.best == _best:
                runner.save_checkpoint(runner.work_dir,filename_tmpl='best.pth',create_symlink=False)

            for idx,name in enumerate(['P', 'R', 'mAP@0.5', 'mAP@0.5:0.95']):
                runner.log_buffer.output[name] = results[idx]
            runner.log_buffer.ready = True

# class DistEvalHook(EvalHook):
#     """Distributed evaluation hook.
#
#     Attributes:
#         dataloader (DataLoader): A PyTorch dataloader.
#         interval (int): Evaluation interval (by epochs). Default: 1.
#         tmpdir (str | None): Temporary directory to save the results of all
#             processes. Default: None.
#         gpu_collect (bool): Whether to use gpu or cpu to collect results.
#             Default: False.
#     """
#
#     def __init__(self,dataloader,interval=1,gpu_collect=False,**eval_kwargs):
#         if not isinstance(dataloader, DataLoader):
#             raise TypeError('dataloader must be a pytorch DataLoader, but got {}'.format(type(dataloader)))
#         self.dataloader = dataloader
#         self.interval = interval
#         self.gpu_collect = gpu_collect
#         self.eval_kwargs = eval_kwargs
#
#     def after_train_epoch(self, runner):
#         if not self.every_n_epochs(runner, self.interval):
#             return
#         from yolodet.apis.test import multi_gpu_test
#         results = multi_gpu_test(
#             runner.model,
#             self.dataloader,
#             tmpdir=osp.join(runner.work_dir, '.eval_hook'),
#             gpu_collect=self.gpu_collect)
#         if runner.rank == 0:
#             print('\n')
#             self.evaluate(runner, results)
