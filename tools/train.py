#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 16:58
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: train.py
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
import copy
import os.path as osp
import time
import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from yolodet.utils.config import Config
from tools import file_utils
from yolodet.utils.Logger import Logging
from yolodet.utils.collect_env import collect_env
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.utils.registry import DETECTORS,DATASET
from yolodet.apis.train import set_random_seed,train_detector
from yolodet.models.utils.torch_utils import select_device



def parse_args():
    parser = argparse.ArgumentParser(description='YOLODet train detectors')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic',action='store_true',help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--autoscale-lr',action='store_true',help='automatically scale lr with the number of gpus')
    parser.add_argument('--multi-scale',action='store_true',help='vary img-size +/- 50%%')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.device is not None:
        cfg.device = args.device
    else:
        cfg.device = None
    device = select_device(cfg.device)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8


    # create work_dir
    file_utils.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = Logging.getLogger()

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +dash_line)
    meta['env_info'] = env_info
    meta['batch_size'] = cfg.data.batch_size
    meta['subdivisions'] = cfg.data.subdivisions
    meta['multi_scale'] = args.multi_scale
    # log some basic info
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    model = build_from_dict(cfg.model, DETECTORS)

    model = model.cuda(device)
    # model.device = device
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.device = device

    datasets = [build_from_dict(cfg.data.train, DATASET)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_from_dict(val_dataset, DATASET))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    train_detector(model,datasets,cfg,validate=args.validate,timestamp=timestamp,meta=meta)


if __name__ == '__main__':
    main()
