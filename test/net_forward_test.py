#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/20 9:52
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: net_forward_test.py 
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
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from yolodet.apis.inference import inference_detector,init_detector,show_result,show_result_pyplot
import os
import torch

image_path = '/disk2/project/mmdetection/mount/hand/ttt/25.jpg'
I = Image.open(image_path).convert('RGB')
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
size = 608

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

tensor = transform(I).unsqueeze(0).requires_grad_()

# model = models.alexnet(pretrained=True)
config = '/disk2/project/pytorch-YOLOv4/cfg/yolov4_hand_gpu.py'
checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/test/epoch_6.pth'
model = init_detector(config, checkpoint=checkpoint, device='cuda:0')
visualisation = {}
def hook_fn(m, i, o):
    visualisation[m] = o

def get_all_layers(model):
    for name, layer in model._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_forward_hook(hook_fn)

def normalize(I):
    # 归一化梯度map，先归一化到 mean=0 std=1
    norm = (I - I.mean()) / I.std()
    # 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
    norm = norm * 0.1
    # 均值加 0.5，保证大部分的梯度值为正
    norm = norm + 0.5
    # 把 0，1 以外的梯度值分别设置为 0 和 1
    norm = norm.clip(0, 1)
    # norm = norm*255
    return norm

get_all_layers(model)
data = dict(img=tensor)
out = inference_detector(model,img=image_path)

layers = [0,1,2]

for index, k in enumerate(visualisation.items()):
    v_ = k[1]
    for L in layers:
        value = v_[L]
        if isinstance(value,torch.Tensor):
            value = value.detach().cpu().numpy()

        if len(value.shape)==1:
            continue
        batch, deep, w, h = value.shape
        print('L={},w={},h={}'.format(L,w, h))
        for i in range(batch):
            for j in range(deep):
                if j>10:
                    break
                v = value[i][j]
                # v = v[:,:,np.newaxis]
                v = normalize(v)
                plt.imshow(v, cmap=plt.cm.gray)
                # plt.show()
                plt.savefig('hand/'+ str(index)+ '_'+str(L) + '_' + str(i) + '_' + str(j) + '.jpg')


# for layer in layers:
#     value = visualisation[list(visualisation.keys())[layer]]
#     value = value.detach().numpy()
#     batch,deep,w,h = value.shape
#     print('w={},h={}'.format(w,h))
#     for i in range(batch):
#         for j in range(deep):
#             v = value[i][j]
#             # v = v[:,:,np.newaxis]
#             v = normalize(v)
#             plt.imshow(v,cmap=plt.cm.gray)
#             # plt.show()
#             plt.savefig('long/'+str(layer)+'_'+str(i)+'_'+str(j)+'.jpg')
            # plt.show()

# result = normalize(visualisation[''])
# plt.imshow(result)
# plt.show()
# print('END')



