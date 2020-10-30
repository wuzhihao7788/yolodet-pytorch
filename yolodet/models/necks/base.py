#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/25 14:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: base.py
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
import torch
import torch.nn.functional as F

from yolodet.models.backbones.base import DarknetConv2D_Norm_Activation


class YOLO_SAM_Module(nn.Module):
    def __init__(self,in_channels,out_channels,norm_type='BN',num_groups=None):
        super(YOLO_SAM_Module,self).__init__()
        self.conv1x1 = DarknetConv2D_Norm_Activation(in_channels,out_channels, kernel_size=1, activation='logistic',norm_type=norm_type,num_groups=num_groups)

    def forward(self,x):
        return self.conv1x1(x)* x


def make_DBL_cluster(in_channels, cluster_num=3,norm_type='BN',num_groups=None,sam=False):
    clusters = []
    assert isinstance(cluster_num, int)
    assert cluster_num in [3, 5]
    for i in range(cluster_num):

        if i % 2 == 1:
            kernel_size = 3
            in_c = in_channels//2
            out_c = in_channels
        else:
            kernel_size = 1
            in_c = in_channels
            out_c = in_channels//2
        if sam and i+1 == cluster_num:
            clusters.append(YOLO_SAM_Module(in_c, in_c,norm_type=norm_type, num_groups=num_groups))
        clusters.append(DarknetConv2D_Norm_Activation(in_c, out_c, kernel_size=kernel_size, activation='leaky',norm_type=norm_type,num_groups=num_groups))
    return clusters

class UpSampleModule(nn.Module):
    def __init__(self,in_channels,norm_type='BN',num_groups=None,sam=False):
        super(UpSampleModule,self).__init__()
        self.conv1x1_1 = DarknetConv2D_Norm_Activation(in_channels, in_channels // 2, kernel_size=1, activation='leaky',norm_type=norm_type,num_groups=num_groups)
        self.conv1x1_2 = DarknetConv2D_Norm_Activation(in_channels, in_channels // 2, kernel_size=1, activation='leaky',norm_type=norm_type,num_groups=num_groups)
        dbl_clusters = make_DBL_cluster(in_channels,cluster_num=5,norm_type=norm_type,num_groups=num_groups,sam=sam)
        self.dbl_cluster_layers = []
        for i ,dbl in enumerate(dbl_clusters):
            layer_name = 'dbl_cluster_layer_{}'.format(i + 1)
            self.add_module(layer_name,dbl)
            self.dbl_cluster_layers.append(layer_name)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x1 = self.conv1x1_1(x1)
        x2 = self.conv1x1_2(x2)
        x1 = F.interpolate(x1, scale_factor=2, mode='nearest')
        out = torch.cat([x1,x2],dim=1)
        for layer_name in self.dbl_cluster_layers:
            dbl_cluster = getattr(self, layer_name)
            out = dbl_cluster(out)
        return out

class DownSampleModule(nn.Module):
    def __init__(self,in_channels,norm_type='BN',num_groups=0,sam=False):
        super(DownSampleModule,self).__init__()
        self.conv3x3 = DarknetConv2D_Norm_Activation(in_channels, in_channels * 2, kernel_size=3, stride=2, activation='leaky',norm_type=norm_type,num_groups=num_groups)
        dbl_clusters = make_DBL_cluster(in_channels*4,cluster_num=5,norm_type=norm_type,num_groups=num_groups,sam=sam)
        self.dbl_cluster_layers = []
        for i ,dbl in enumerate(dbl_clusters):
            layer_name = 'dbl_cluster_layer_{}'.format(i + 1)
            self.add_module(layer_name,dbl)
            self.dbl_cluster_layers.append(layer_name)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x2 = self.conv3x3(x2)
        out = torch.cat([x1,x2],dim=1)
        for dbl_name in self.dbl_cluster_layers:
            dbl = getattr(self,dbl_name)
            out = dbl(out)
        return out


class SPP(nn.Module):
    def __init__(self,kernel_sizes=[5, 9, 13]):

        super(SPP,self).__init__()
        max_pool_cluster = self.make_max_pool_cluster(kernel_sizes)

        self.max_pool_cluster_layers = []
        for i, mpc in enumerate(max_pool_cluster):
            layer_name = 'max_pool_cluster_{}'.format(i + 1)
            self.add_module(layer_name, mpc)
            self.max_pool_cluster_layers.append(layer_name)

    def make_max_pool_cluster(self, kernel_sizes):
        clusters = []
        for kernel_size in kernel_sizes:
            pad = kernel_size // 2
            clusters.append(nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad))
        return clusters

    def forward(self, x):
        outs = []
        for max_pool_name in self.max_pool_cluster_layers:
            max_pool = getattr(self, max_pool_name)
            outs.append(max_pool(x))
        outs.append(x)
        out = torch.cat(outs,dim=1)
        return out


class YOLO_SPP(nn.Module):

    def __init__(self,in_channels,kernel_sizes=[5, 9, 13],norm_type='BN',num_groups=None):
        super(YOLO_SPP,self).__init__()
        dbl_cluster_3_1 = make_DBL_cluster(in_channels, cluster_num=3,norm_type=norm_type,num_groups=num_groups)
        self.dbl_cluster_3_1_layers = []
        for i ,dbl in enumerate(dbl_cluster_3_1):
            layer_name = 'dbl_cluster_3_1_layer_{}'.format(i + 1)
            self.add_module(layer_name,dbl)
            self.dbl_cluster_3_1_layers.append(layer_name)

        dbl_cluster_3_2 = [
            DarknetConv2D_Norm_Activation(in_channels * 2, in_channels // 2, kernel_size=1, activation='leaky',
                                          norm_type=norm_type, num_groups=num_groups),
            DarknetConv2D_Norm_Activation(in_channels // 2, in_channels, kernel_size=3, activation='leaky',
                                          norm_type=norm_type, num_groups=num_groups),
            DarknetConv2D_Norm_Activation(in_channels, in_channels // 2, kernel_size=1, activation='leaky',
                                          norm_type=norm_type, num_groups=num_groups)
        ]

        self.dbl_cluster_3_2_layers = []
        for i, dbl in enumerate(dbl_cluster_3_2):
            layer_name = 'dbl_cluster_3_2_{}'.format(i + 1)
            self.add_module(layer_name, dbl)
            self.dbl_cluster_3_2_layers.append(layer_name)

        max_pool_cluster = self.make_max_pool_cluster(kernel_sizes)

        self.max_pool_cluster_layers = []
        for i ,mpc in enumerate(max_pool_cluster):
            layer_name = 'max_pool_cluster_{}'.format(i + 1)
            self.add_module(layer_name,mpc)
            self.max_pool_cluster_layers.append(layer_name)


    def make_max_pool_cluster(self,kernel_sizes):
        clusters = []
        for kernel_size in kernel_sizes:
            pad = kernel_size//2
            clusters.append(nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=pad))
        return clusters

    def forward(self,x):
        for dbl_name in self.dbl_cluster_3_1_layers:
            dbl = getattr(self,dbl_name)
            x = dbl(x)
        outs = []
        for max_pool_name in self.max_pool_cluster_layers:
            max_pool = getattr(self, max_pool_name)
            outs.append(max_pool(x))
        outs.append(x)
        out = torch.cat(outs,dim=1)
        for dbl_name in self.dbl_cluster_3_2_layers:
            dbl = getattr(self, dbl_name)
            out = dbl(out)
        return out


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=1, with_r=False,activation='leaky',norm_type='BN',num_groups=None,coord_conv=True):
        super().__init__()
        self.coord_conv = coord_conv
        if self.coord_conv:
            self.addcoords = AddCoords(with_r=with_r)
            # in_size = in_channels
            in_channels += 2
            if with_r:
                in_channels += 1
        self.conv = DarknetConv2D_Norm_Activation(in_channels, out_channels, kernel_size=kernel_size, activation=activation,norm_type=norm_type,num_groups=num_groups)


    def forward(self, x):
        if self.coord_conv:
            x = self.addcoords(x)
        # ret = x
        x = self.conv(x)
        return x


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()

        self.drop_prob = 1-keep_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

# if __name__ == '__main__':
#     input = torch.rand(1,3,512,512)
#     ccon = CoordConv(3,6,kernel_size=3)
#     ccon.forward(input)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


