#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/27 10:38
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: custom.py 
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
from collections import defaultdict

from torch.utils.data import Dataset
import os.path as osp
import numpy as np

from yolodet.dataset.pipelines.compose import Compose


class Custom(Dataset):

    CLASSES = None

    def __init__(self, ann_file,name_file, data_root, img_prefix,pipeline,num_bbox=60, train=True):
        super(Custom, self).__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.name_file = name_file
        self.train = train
        if num_bbox is not None:
            self.num_bbox = num_bbox
        else:
            self.num_bbox = None

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not osp.isabs(self.name_file):
                self.name_file = osp.join(self.data_root, self.name_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
        # truth[imgname] = [x1,y1,x2,y2,label] 未格式化
        self.truth = self.load_annotations(self.ann_file)
        self.CLASSES = self.load_names(self.name_file)

        self.imgs = list(self.truth.keys())
        self.pipeline = Compose(pipeline)


    def __len__(self):
        return len(self.truth.keys())

    def load_annotations(self, ann_file):
        truth = defaultdict(list)
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = line.split(" ")
                image = data[0]
                for i in data[1:]:
                    truth[image].append([int(j) for j in i.split(',')])
        return truth

    def load_names(self, name_file):
        names = []
        f = open(name_file, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.strip()
            names.append(data)
        return names

    def prepare_test_img(self, idx):
        img_ = self.imgs[idx]
        truth_ = self.truth[img_]
        results = dict(y_true=truth_, img_name=img_)

        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['truth'] = self.truth
        results['img_list'] = self.imgs
        if self.num_bbox is not None:
            results['num_bbox'] = self.num_bbox

    def prepare_train_img(self, idx):
        img_ = self.imgs[idx]
        truth_ = self.truth[img_]
        results = dict(y_true=truth_, img_name=img_)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        if not self.train:
            return self.prepare_test_img(idx)

        data = self.prepare_train_img(idx)
        return data

    def get_ann_info(self,idx):
        return_dict = dict()
        img_ = self.imgs[idx]
        truth_ = self.truth[img_]
        bboxes = []
        labels = []
        for i,yt in enumerate(truth_):
            labels.append(yt[4])
            bboxes.append(yt[:4])
        return_dict['bboxes'] = np.array(bboxes, dtype=np.float32)
        return_dict['labels'] = np.array(labels, dtype=np.int64)
        return_dict['img'] = img_
        # return_dict['bboxes_ignore'] = np.array([], dtype=np.float32)
        # return_dict['labels_ignore'] = np.array([], dtype=np.int64)
        return return_dict
