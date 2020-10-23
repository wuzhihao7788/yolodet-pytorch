#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# copy and modify from mmcv.runner.hooks.checkpoint_hook.py
# @Time    : 2020/8/10 16:27
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: checkpoint_hook.py
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
from .hook import Hook

class CheckpointHook(Hook):

    def __init__(self,interval=-1,save_optimizer=True,out_dir=None,**kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)