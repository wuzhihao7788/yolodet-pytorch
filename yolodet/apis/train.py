#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 16:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: train_bk.py
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
import math
import random
from collections import OrderedDict

from yolodet.models.utils.torch_utils import select_device
from yolodet.utils.Logger import Logging
import torch.nn.functional as F

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from yolodet.dataset.loader.build_dataloader import build_dataloader
from yolodet.apis.runner import Runner
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.utils.registry import DATASET


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value if _loss is not None)
        # else:
        #
        #     raise TypeError(
        #         '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key and 'Y' not in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.detach().item()

    return loss, log_vars


def batch_processor(model, data, train_mode,**kwargs):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    if 'multi_scale' in kwargs and kwargs['multi_scale']:
    # if 'multi_scale' in kwargs:
        imgs = data['img']
        imgsz = imgs.shape[-1]
        # sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + 32) // 32 * 32  # size
        sz = random.randrange(imgsz * 0.5, imgsz + 32) // 32 * 32  # size
        sf = sz / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [math.ceil(x * sf / 32) * 32 for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
            data['img'] = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            for img_metas in data['img_metas']:
                img_metas['img_shape'] = ns[0], ns[0]
    data['img'] = data['img'].to(model.device)
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss*len(data['img']), log_vars=log_vars, num_samples=len(data['img']))

    return outputs


def train_detector(model,dataset,cfg,validate=False,timestamp=None,meta=None):
    logger = Logging.getLogger()
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(ds,data=cfg.data) for ds in dataset
    ]

    # build runner
    optimizer = cfg.optimizer

    if 'ema' in cfg:
        ema = cfg.ema
    else:
        ema = None
    runner = Runner(model,batch_processor,optimizer,cfg.work_dir,logger=logger,meta=meta,ema=ema)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks 需要放在日志前面，不然打印不出日志。
    if validate:
        cfg.data.val.train=False
        val_dataset = build_from_dict(cfg.data.val, DATASET)
        val_dataloader = build_dataloader(val_dataset,shuffle=False,data=cfg.data)
        eval_cfg = cfg.get('evaluation', {})
        from yolodet.models.hooks.eval_hook import EvalHook
        runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

    # register hooks
    # runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,cfg.checkpoint_config)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
