#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: Darknet.py
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

from yolodet.models.backbones.base import CSP_ResBock_Body, DarknetConv2D_Norm_Activation

from yolodet.models.utils.torch_utils import initialize_weights

arch_settings = {
    53: (CSP_ResBock_Body, (1, 2, 8, 8, 4)),
}



class CSPDarknet(nn.Module):
    def __init__(self, in_channels=3,planes=32, out_indices=(2, 3, 4,), depth=53,norm_type='BN',num_groups=32):
        super(CSPDarknet,self).__init__()
        assert isinstance(depth, int)
        assert norm_type in ('BN','GN'),'norm_type must BN or GN'
        if depth not in arch_settings.keys():
            depth = 53
        self.out_indices = out_indices

        self.norm_type = norm_type
        self.num_groups = num_groups
        self.conv3x3 = DarknetConv2D_Norm_Activation(in_channels, planes, kernel_size=3, activation='mish',norm_type=self.norm_type,num_groups=self.num_groups)

        self.block, self.stage_blocks = arch_settings[depth]

        self.csp_res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            _planes = planes * 2**i
            layer_name = 'csp_res_layer_{}'.format(i + 1)
            self.add_module(layer_name, self.block(in_channels=_planes, res_num=num_blocks,norm_type=self.norm_type,num_groups=self.num_groups))
            self.csp_res_layers.append(layer_name)

        self.init_weights()

    def forward(self, x):
        outs = []
        out = self.conv3x3(x)
        for i, layer_name in enumerate(self.csp_res_layers):
            res_block = getattr(self, layer_name)
            out = res_block(out)
            if i in self.out_indices:
                outs.append(out)
        if len(outs) == 0:
            outs.append(out)

        return outs

    def init_weights(self,pretrained=None):
        initialize_weights(self)