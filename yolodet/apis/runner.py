#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 10:37
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: runner.py
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
#copy and modify from mmcv.runner.runner

# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import time
from collections import OrderedDict
from getpass import getuser
from socket import gethostname

from yolodet.apis.log_buffer import LogBuffer
from yolodet.models.hooks.hook import Hook
from yolodet.models.hooks.iter_timer_hook import IterTimerHook
from yolodet.models.utils.torch_utils import ModelEMA
from yolodet.utils.registry import HOOKS
from yolodet.utils.newInstance_utils import obj_from_dict,build_from_dict
from tools import file_utils
from yolodet.utils.Logger import Logging
from yolodet.apis.checkpoint import load_checkpoint,save_checkpoint

import torch



class Runner(object):
    def __init__(self,model,batch_processor,optimizer=None,work_dir=None,logger=None,meta=None,ema=None):
        assert callable(batch_processor)
        self.model = model
        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self._warmup_max_iters = 0
        self._momentum = 0

        self.batch_processor = batch_processor

        # create work_dir
        if isinstance(work_dir,str):
            self.work_dir = osp.abspath(work_dir)
            file_utils.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        if logger is None:
            self.logger = Logging.getLogger()
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None

        if ema is not None:
            self.ema = self.init_ema(ema)
        else:
            self.ema = None

    @property
    def momentum(self):
        """str: Name of the model, usually the module class name."""
        return self._momentum

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch
    @property
    def warmup_max_iters(self):
        """int: Current epoch."""
        return self._warmup_max_iters
    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_ema(self,ema):
        return ModelEMA(self.model,**ema)

    def init_optimizer(self, optimizer):

        if isinstance(optimizer, dict):
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    if '.bias' in k:
                        pg2.append(v)  # biases
                    elif '.weight' in k and '.bn' not in k:
                        pg1.append(v)  # apply weight decay 注释：conv的权重
                    else:
                        pg0.append(v)  # all else BN的权重

            if 'weight_decay' in optimizer.keys():
                weight_decay = optimizer.pop('weight_decay')
            else:
                weight_decay = 5e-4

            if 'momentum' in optimizer.keys():
                self._momentum = optimizer['momentum']

            optimizer = obj_from_dict(optimizer, torch.optim,dict(params=pg0))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

        return optimizer

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            self._hooks.insert(i + 1, hook)
            inserted = True
            break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)


    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if 'multi_scale' in self.meta:
                kwargs['multi_scale'] = self.meta['multi_scale']
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')

            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location=map_location, strict=strict)

    def resume(self,checkpoint,resume_optimizer=True,map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(self.model,checkpoint,map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(self.model,checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def save_checkpoint(self,out_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True, meta=None, create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        if self.ema is not None:
            model = self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema
            save_checkpoint(model, filepath, optimizer=optimizer, meta=meta)
        else:
            save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        try:
            if create_symlink:
                file_utils.symlink(filepath, osp.join(out_dir, 'latest.pth'))
        except:
            self.logger.warning('create_symlink failed')

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert isinstance(workflow,list)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s','{}@{}'.format(getuser(), gethostname()), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            hook_type = lr_config.pop('policy').title() + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = build_from_dict(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_dict(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = build_from_dict(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = build_from_dict(info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_lr_hook(lr_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)