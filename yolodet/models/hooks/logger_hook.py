#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/11 18:09
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: logger_hook.py
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
from abc import ABCMeta,abstractmethod
from yolodet.models.hooks.hook import Hook
import datetime
import os.path as osp
from collections import OrderedDict

import torch

class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

class TextLoggerHook(LoggerHook):

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)
        self.start_iter = runner.iter
        self.json_log_path = osp.join(runner.work_dir,
                                      '{}.log.json'.format(runner.timestamp))
        if runner.meta is not None:
            self._dump_log(runner.meta, runner)

    def _get_max_memory(self, runner):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=torch.device('cuda'))
        return mem_mb.item()

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            log_str = 'Epoch [{}][{}/{}]\tlr: {:.5f}, '.format(
                log_dict['epoch'], log_dict['iter'], len(runner.data_loader),
                log_dict['lr'])
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += 'eta: {}, '.format(eta_str)
                log_str += ('time: {:.3f}, data_time: {:.3f}, '.format(
                    log_dict['time'], log_dict['data_time']))
                log_str += 'memory: {}, '.format(log_dict['memory'])
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(log_dict['mode'],
                                                    log_dict['epoch'] - 1,
                                                    log_dict['iter'])
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    def _dump_log(self, log_dict, runner):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        # if runner.rank == 0:
        with open(self.json_log_path, 'a+') as f:
            json.dump(json_log, f)
            f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, runner):
        log_dict = OrderedDict()
        # training mode if the output contains the key "time"
        mode = 'train' if 'time' in runner.log_buffer.output else 'val'
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        # only record lr of the first param group
        log_dict['lr'] = runner.current_lr()[0]
        if mode == 'train':
            log_dict['time'] = runner.log_buffer.output['time']
            log_dict['data_time'] = runner.log_buffer.output['data_time']
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)
        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_dict[name] = val

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)


class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir

    def before_run(self, runner):
        if torch.__version__ >= '1.1' and '.' in torch.__version__:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)

    def after_run(self, runner):
        self.writer.close()
