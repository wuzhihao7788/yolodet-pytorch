#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# copy and update from https://github.com/4uiiurz1/pytorch-deform-conv-v2/blob/master/deform_conv_v2.py
# @Time    : 2020/8/31 18:19
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: DeformConv
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

import torch
from torch import nn
import os
# CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.conv_offset = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        # self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # self.m_conv.register_backward_hook(self._set_lr)

    # @staticmethod
    # def _set_lr(module, grad_input, grad_output):
    #     grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
    #     grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    """
    可变性卷积的流程为：

    1、原始图片batch（大小为b*h*w*c），记为U，经过一个普通卷积，卷积填充为same，即输出输入大小不变，
    对应的输出结果为（b*h*w*2c)，记为V，输出的结果是指原图片batch中每个像素的偏移量（x偏移与y偏移，因此为2c）。

    2、将U中图片的像素索引值与V相加，得到偏移后的position（即在原始图片U中的坐标值），需要将position值限定为图片大小以内。
    position的大小为（b*h*w*2c)，但position只是一个坐标值，而且还是float类型的，我们需要这些float类型的坐标值获取像素。

    3、取一个坐标值（a,b)，将其转换为四个整数，floor(a), ceil(a), floor(b), ceil(b)，将这四个整数进行整合，
    得到四对坐标（floor(a),floor(b)),  ((floor(a),ceil(b)),  ((ceil(a),floor(b)),  ((ceil(a),ceil(b))。这四对坐标每个坐标都对应U
    中的一个像素值，而我们需要得到(a,b)的像素值，这里采用双线性差值的方式计算（一方面得到的像素准确，另一方面可以进行反向传播）。

    4、在得到position的所有像素后，即得到了一个新图片M，将这个新图片M作为输入数据输入到别的层中，如普通卷积。
    
    """
    def forward(self, x):
        if len(x[torch.isnan(x)])>0:
            print(x)
        offset = self.conv_offset(x)#卷积核（3x3）x,y 的偏移量（18）
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()#左上(向下取整)
        q_rb = q_lt + 1#右下

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()#left top 左上
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()#right buttom 右下
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)#left buttom 左下
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)#right top 右上

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        if len(q_lt[torch.lt(q_lt,0)])>0:
            print(q_lt)
        if len(q_rb[torch.lt(q_rb,0)])>0:
            print(q_rb)
        if len(q_lb[torch.lt(q_lb,0)])>0:
            print(q_lb)
        if len(q_rt[torch.lt(q_rt,0)])>0:
            print(q_rt)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)

        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        # p_n = p_n.contiguous().view(1, 2*N, 1, 1)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset.device)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype).to(offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        if len(index[torch.lt(index,0)])>0:
            print(index)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


# if __name__ == '__main__':
#     input = torch.rand(1,3,512,512)
#     dcn = DeformConv2d(3,6)
#     dcn.forward(input)