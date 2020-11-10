简体中文 | [English](README_en.md)
# YOLODet-PyTorch
YOLODet-PyTorch是端到端基于pytorch框架复现yolo最新算法的目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的训练、精度速度优化到部署全流程。YOLODet-PyTorch以模块化的设计实现了多种主流YOLO目标检测算法，并且提供了丰富的数据增强、网络组件、损失函数等模块。

**目前检测库下模型均要求使用PyTorch 1.5及以上版本或适当的develop版本。**
## 内容
- [简介](#简介)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [预训练模型](#预训练模型)
- [重要说明](#重要说明)
- [鸣谢](#鸣谢)
- [如何贡献代码](#如何贡献代码)

## 简介

### 特性：

- 模型丰富：

  YOLODet提供了丰富的模型，涵盖最新YOLO检测算法的复现，包含YOLOv5、YOLOv4、PP-YOLO、YOLOv3等YOLO系列目标检测算法。

- 高灵活度：

  YOLODet通过模块化设计来解耦各个组件，基于配置文件可以轻松地搭建各种检测模型。

### 支持的模型:

- [YOLOv5(s,m,l,x)](docs/yolov5_cn.md)
- [YOLOv4(标准版，sam版)](docs/yolov4_cn.md) 
- [PP-YOLO](docs/pp-yolo_cn.md)
- [YOLOv3](docs/yolov3_cn.md)

### 更多的Backone：

- DarkNet
- CSPDarkNet
- ResNet
- YOLOv5Darknet

### 数据增强方法：

- Mosaic
- MixUp
- Resize
- LetterBox
- RandomCrop
- RandomFlip
- RandomHSV
- RandomBlur
- RandomNoise
- RandomAffine
- RandomTranslation
- Normalize
- ImageToTensor
- 相关配置使用说明请参考【[这里](docs/TRANSFORMS_cn.md)】

### 损失函数支持：

- bbox loss (IOU,GIOU,DIOU,CIOU)
- confidence loss(YOLOv4,YOLOv5,PP-YOLO)
- IOU_Aware_Loss(PP-YOLO)
- FocalLoss


### 训练技巧支持：

- [指数移动平均](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- 预热
- 梯度剪切
- 梯度累计更新
- 多尺度训练
- 学习率调整：Fixed，Step，Exp，Poly，Inv，Consine
- Label Smooth
- **强烈说明** 通过实验对比发现YOLOv5的正负样本划分定义和损失函数定义，使得模型收敛速度较快，远超原yolo系列对正负样本的划分和损失定义。对于如果卡资源不充足，想在短时间内收敛模型，可采用yolov5的正负样本划分和损失函数定义，相关参数为`yolo_loss_type=yolov5`。
- 额外补充 YOLOv5对于正样本的定义:在不同尺度下只要真框和给定锚框的的比值在4倍以内，该锚框即可负责预测该真值框。并根据gx,gy在grid中心点位置的偏移量会额外新增两个grid坐标来预测。通过这一系列操作，增加了正样本数量，加速模型收敛速度。而YOLO原系列对于真框，在不同尺度下只有在该尺度下IOU交并集最大的锚框负责预测该真框，其他锚框不负责，所以由于较少的正样本量，模型收敛速度较慢。

### 扩展特性：

- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Focus**
- [x] **Spatial Pyramid Pooling**
- [x] **FPN-PAN**
- [x] **coord conv**
- [x] **drop block**
- [x] **SAM**


### 代码结构说明
```
yolodet-pytorch
├──cfg              #模型配置文件所在目录(yolov5,yolov4等)
├──tools            #工具包，包含训练代码，测试代码和推断代码入口。
├──yolodet          #YOLO检测框架核心代码库
│  ├──apis          #提供检测框架的训练，测试和推断和模型保存的接口
│  ├──dataset       #包含DateSet，DateLoader和数据增强等通用方法
│  ├──models        #YOLO检测框架的核心组件集结地
│  │  ├──detectors  #所有类型检测器集结地
│  │  ├──backbones  #所有骨干网络集结地
│  │  ├──necks      #所有necks集结地
│  │  ├──heads      #heads集结地
│  │  ├──loss       #所有损失函数集结地
│  │  ├──hooks      #hooks集结地(学习率调整，模型保存，训练日志，权重更新等)
│  │  ├──utils      #所有工具方法集结地
```

## 安装说明

安装和数据集准备请参考 [INSTALL.md](docs/INSTALL_cn.md) 。


## 快速开始

请参阅 [GETTING_STARTED.md](docs/GETTING_STARTED_cn.md) 了解YOLODet的基本用法。

## 预训练模型
查看预训练模型请点击【[这里](docs/MODEL_ZOO_cn.md)】

## 重要说明
由于该检测框架为个人闲暇之余，处于对深度学习的热爱，自己单独编写完成，也由于自己囊中羞涩，没有充足的显卡资源，提供的MSCOCO大型数据集的预训练模型为未完整训练的模型，后面会陆陆续续提供完整版的预训练模型，敬请大家期待。对于小型数据集本人已经测试和验证过，并在实际项目中使用过该框架训练的模型，没有问题，精度和速度都能保证。

## 鸣谢
- 本检测框架中的代码结构参考[open-mmlab](https://github.com/open-mmlab)的 [mmdetection](https://github.com/open-mmlab/mmdetection)并做部分引用，已在代码注释中说明。
- 神经网络网络结构可视化工具：Netron https://github.com/lutzroeder/Netron
- Paper YOLOv4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
- YOLOv5：https://github.com/ultralytics/yolov5
- PP-YOLO：https://arxiv.org/abs/2007.12099
- PP-YOLO code：https://github.com/PaddlePaddle/PaddleDetection


## 如何贡献代码

欢迎你为YOLODet提供代码，也十分感谢你的反馈。本人将不断完善和改进这个基于PyTorch实现的YOLO全系列的目标检测框架，并希望更多志同道合的爱好者和从业者能参与进来，共同维护这个项目。
如有对此感兴趣的同学，可联系我的gmail邮箱:wuzhihao7788@gmail.com,期待与你一起完善和进步。
