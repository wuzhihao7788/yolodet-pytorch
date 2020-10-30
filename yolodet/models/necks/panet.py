#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: panet.py
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
import torch.nn as nn

from yolodet.models.necks.base import UpSampleModule, DownSampleModule, YOLO_SPP

from yolodet.models.utils.torch_utils import initialize_weights


class PANet(nn.Module):

    # def __init__(self, in_channels = 1024,kernel_sizes = [5, 9, 13]):
    def __init__(self, in_channels = [256,512,1024],kernel_sizes = [5, 9, 13],norm_type='BN',num_groups=None,sam=False):
        super(PANet,self).__init__()
        self.spp = YOLO_SPP(in_channels[-1],kernel_sizes=kernel_sizes,norm_type=norm_type,num_groups=num_groups)# out_channels = 512
        self.upsample1 = UpSampleModule(in_channels[-2],norm_type=norm_type,num_groups=num_groups) # out_channels = 256
        self.upsample2 = UpSampleModule(in_channels[-3],norm_type=norm_type,num_groups=num_groups,sam=sam) # out_channels = 128
        self.downsample1 = DownSampleModule(in_channels[-2]//2,norm_type=norm_type,num_groups=num_groups,sam=sam)
        self.downsample2 = DownSampleModule(in_channels[-3]//2,norm_type=norm_type,num_groups=num_groups,sam=sam)
        #in_channels = 1024
        # self.spp = SPP(in_channels,kernel_sizes=kernel_sizes)# out_channels = 512
        # self.upsample1 = UpSampleModule(in_channels//2) # out_channels = 256
        # self.upsample2 = UpSampleModule(in_channels//4) # out_channels = 128
        # self.downsample1 = DownSampleModule(in_channels//4)
        # self.downsample2 = DownSampleModule(in_channels//8)

    def forward(self,x):
        #x.size = 3 The last three layers
        x1 = self.spp(x[-1]) # The third
        x2 = x[1] #The second
        x3 = x[0] #The first
        x2 = self.upsample1([x1,x2]) # third and second upsample to second
        x3 = self.upsample2([x2,x3]) # second and first upsample to first
        x2 = self.downsample2([x2,x3])  # second and first downsample to second
        x1 = self.downsample1([x1,x2]) # second and third downsample to third
        return [x1,x2,x3]

    def init_weights(self):
        initialize_weights(self)