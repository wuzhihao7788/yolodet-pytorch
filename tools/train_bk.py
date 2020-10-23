#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/29 18:19
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: train_bk.py
# @Software: PyCharm
# @github ：https://github.com/wuzhihao7788/pytorch-YOLOv4

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
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.data.dataloader import default_collate
import torch
import collections

from yolodet.utils.newInstance_utils import build_from_dict
import os
from yolodet.utils.registry import DETECTORS,DATASET
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Trainer(object):

    def __init__(self, batch=8, subdivisions=4,epochs=100, burn_in=1000, steps=[400000, 450000]):

        _model = build_from_dict(model, DETECTORS)
        self.model = DataParallel(_model.cuda(), device_ids=[0])

        self.train_dataset = build_from_dict(data_cfg['train'], DATASET)
        self.val_dataset = build_from_dict(data_cfg['val'], DATASET)

        self.burn_in = burn_in
        self.steps = steps
        self.epochs = epochs

        self.batch = batch
        self.subdivisions = subdivisions

        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch // subdivisions, shuffle=True,
                                       num_workers=1, pin_memory=True, drop_last=True, collate_fn=self.collate)

        self.val_loader = DataLoader(self.val_dataset, batch_size=batch // subdivisions, shuffle=True,
                                     num_workers=1,
                                     pin_memory=True, drop_last=True, collate_fn=self.collate)

        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001 / batch,betas=(0.9, 0.999),eps=1e-08,)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.burnin_schedule)

    def train(self):
        self.model.train()
        global_step = 0
        checkpoints = r'/disk2/project/pytorch-YOLOv4/checkpoints/'
        save_prefix = 'Yolov4_epoch_'
        saved_models = collections.deque()
        for epoch in range(self.epochs):

            epoch_loss = 0
            epoch_step = 0

            for i, batch in enumerate(self.train_loader):
                losses = self.model(**batch)
                loss = self.parse_losses(losses)
                loss.backward()
                epoch_loss += loss.item()
                print('loss :{}'.format(loss))

                global_step += 1
                epoch_step += 1

                if global_step % self.subdivisions == 0:
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    self.scheduler.step()

            try:
                # os.mkdir(config.checkpoints)
                os.makedirs(checkpoints, exist_ok=True)
            except OSError:
                pass
            save_path = os.path.join(checkpoints, f'{save_prefix}{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)

            saved_models.append(save_path)
            if len(saved_models) > 5:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    pass



    def burnin_schedule(self, i):
        if i < self.burn_in:
            factor = pow(i / self.burn_in, 4)
        elif i < self.steps[0]:
            factor = 1.0
        elif i < self.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    def collate(self, batch):
        if 'multi_scale' in data_cfg.keys() and len(data_cfg['multi_scale'])>0:
            multi_scale = data_cfg['multi_scale']
            if isinstance(multi_scale,dict) and 'type' in multi_scale.keys():
                randomShape = build_from_dict(multi_scale, TRANSFORMS)
                batch = randomShape(batch)
        collate = default_collate(batch)
        return collate

    def parse_losses(self,losses):
        log_vars = collections.OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        return loss

if __name__ == '__main__':
    train = Trainer()
    train.train()