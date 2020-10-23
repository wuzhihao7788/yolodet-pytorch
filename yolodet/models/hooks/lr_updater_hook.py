#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# copy and modify from mmcv.runner.hooks.lr_updater.py
# @Time    : 2020/8/10 15:27
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: lr_updater_hook.py 
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
from math import cos, pi
import numpy as np

from .hook import Hook


class LrUpdaterHook(Hook):
    """LR Scheduler in MMCV

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,by_epoch=True,warmup=None,warmup_iters=0,warmup_ratio=0.1,warmup_by_epoch=False,**kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr
            if 'momentum' in param_group:
                if runner.momentum<=0.9:
                    momentum = 0.9
                else:
                    momentum = runner.momentum
                param_group['momentum'] = np.interp(runner.iter, [0,self.warmup_iters], [0.9, momentum])

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if self.warmup_by_epoch:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len

        runner._warmup_max_iters = self.warmup_iters

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


class StepLrUpdaterHook(LrUpdaterHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)

#https://arxiv.org/pdf/1812.01187.pdf
class CosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))