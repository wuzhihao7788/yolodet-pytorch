#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 17:45
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: build_dataloader.py 
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
import os
import platform
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data import DataLoader
from yolodet.utils.registry import TRANSFORMS
from yolodet.utils.newInstance_utils import build_from_dict

from torch.utils.data.dataloader import default_collate

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataloader(dataset,data,shuffle=True,**kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    batch_size = data.batch_size//data.subdivisions
    num_workers = min([os.cpu_count() // data.workers_per_gpu, batch_size if batch_size > 1 else 0, 8])
    # num_workers = data.workers_per_gpu
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=False,
        drop_last=True,
        **kwargs)

    return data_loader
import numpy as np
def collate(batch):
    if len(batch)==0:
        return None
    clt = defaultdict(list)
    for i,dic in enumerate(batch):
        clt['idx'].append(torch.tensor(i))
        for k,v in dic.items():
            clt[k].append(v)

    for k,v in clt.items():
        if isinstance(clt[k][0],torch.Tensor):
            clt[k] = torch.stack(v, 0)
    # collate = default_collate(batch)
    return clt