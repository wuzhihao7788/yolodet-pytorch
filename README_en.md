[简体中文](README.md) | English
# YOLODet-PyTorch
YOLODet-PyTorch is an end-to-end object detector development kit based on the pytorch framework to reproduce the latest Yolo algorithm. It aims to help developers complete the whole process of detection model training, accuracy and speed optimization, and deployment faster and better. YOLODet-PyTorch implements a variety of mainstream YOLO target detection algorithms with a modular design, and provides a wealth of modules such as data enhancement, network components, and loss functions.

**Currently, all models in the detection library require the use of PyTorch 1.5 and above or the appropriate develop version.**  

## Content
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Quick Start](#Quick-Start)
- [Pretraining Model](#Pretraining-Model)
- [Important](#Important)
- [Thanks](#Thanks)
- [How To Contribute Code](#How-To-Contribute-Code)

## Introduction

### Features:

- Rich models:

   YOLODet provides a wealth of models, covering the reproduction of the latest YOLO detection algorithms, include YOLO series target detection algorithms such as YOLOv5, YOLOv4, PP-YOLO, YOLOv3, etc.

- High flexibility:

  YOLODet decouples various components through modular design, and can easily build various detection models based on configuration files.

### Supported models:

- [YOLOv5(s,m,l,x)](docs/yolov5.md)
- [YOLOv4(standard,SAM)](docs/yolov4.md)
- [PP-YOLO](docs/pp-yolo.md)
- [YOLOv3](docs/yolov3.md)

### More Backone:

- DarkNet
- CSPDarkNet
- ResNet
- YOLOv5Darknet

### Data enhancement method:

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
- Please refer to [[here](docs/TRANSFORMS.md)] for related configuration instructions

### Loss function support:

- bbox loss (IOU,GIOU,DIOU,CIOU)
- confidence loss (YOLOv4, YOLOv5, PP-YOLO)
- IOU_Aware_Loss(PP-YOLO)
- FocalLoss


### Training skills support:

- [Exponential Moving Average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- Warm up
- Gradient shear
- Gradient cumulative update
- Multi-scale training
- Learning rate adjustment: Fixed, Step, Exp, Poly, Inv, Consine
- Label Smooth
- **Strong explanation** Through experimental comparison, it is found that YOLOv5's positive and negative sample division definition and loss function definition make the model converge faster, far exceeding the original yolo series' division and loss definition of positive and negative samples. If the card resources are insufficient and you want to converge the model in a short time, you can use yolov5's positive and negative sample partition and loss function definition, the relevant parameter is `yolo_loss_type=yolov5`.
- Additional supplement YOLOv5's definition of positive samples: As long as the ratio of the true frame to the given anchor frame is within 4 times at different scales, the anchor frame can be responsible for predicting the true value frame. And according to the offset of gx and gy in the center of the grid, two additional grid coordinates will be added to predict. Through this series of operations, the number of positive samples is increased and the model convergence speed is accelerated. For the original YOLO series, for the true frame, only the anchor frame with the largest IOU intersection at this scale is responsible for predicting the true frame at different scales, and other anchor frames are not responsible. Therefore, due to the smaller positive sample size, the model converges faster. slow.

### Extended features:

- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Focus**
- [x] **Spatial Pyramid Pooling**
- [x] **FPN-PAN**
- [x] **coord conv**
- [x] **drop block**
- [x] **SAM**


### Code structure description
```
yolodet-pytorch
├──cfg              #The directory where the model configuration file is located (yolov5, yolov4, etc.)
├──tools            #toolkit, contains training code, test code and inference code entry.
├──yolodet          #YOLO detection framework core code base
│ ├──apis           #Provide an interface for training, testing and inference of the detection framework and model preservation
│ ├──dataset        #Contains general methods such as DateSet, DateLoader and data enhancement
│ ├──models         #The core components of the YOLO detection framework are assembled
│ │ ├──detectors    #Assembly of all types of detectors
│ │ ├──backbones    #All backbone network gathering places
│ │ ├──necks        #All necks gathering place
│ │ ├──heads        #heads assembly place
│ │ ├──loss         #The gathering place of all loss functions
│ │ ├──hooks        #hooks assembly place (learning rate adjustment, model saving, training log, weight update, etc.)
│ │ ├──utils        #All tools and methods gathering place
```

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and data set preparation.


## Quick start

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for basic usage of YOLODet.

## Pretraining Model

Please click 【[here](docs/MODEL_ZOO.md)】 to view the pre-training model.

## Important
Due to the testing framework for personal leisure, in the love of deep learning, written by herself, but also because their cash-strapped, do not have enough graphics resources, provide the MSCOCO large data sets of training model for the complete training model, behind will provide a complete version of the training model of the hall, please look forward to.For the model with small data sets that I have tested and verified and that I have used the framework to train in the actual project, there is no problem, and the accuracy and speed can be guaranteed.

## Thanks
- Refer to [mmdetection](https://github.com/open-mmlab/mmdetection) of [open-mmlab](https://github.com/open-mmlab) for the code structure in this detection framework and make some references , Has been explained in the code comments.
- Neural network structure visualization tool: Netron https://github.com/lutzroeder/Netron
- Paper YOLOv4: https://arxiv.org/abs/2004.10934
- Source code: https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
- YOLOv5: https://github.com/ultralytics/yolov5
- PP-YOLO: https://arxiv.org/abs/2007.12099
- PP-YOLO code: https://github.com/PaddlePaddle/PaddleDetection


## How To Contribute Code

You are welcome to provide the code for YOLODet and your feedback is greatly appreciated.I will constantly improve and improve the object detection framework of the full series of YOLO based on PyTorch, and I hope more like-minded enthusiasts and practitioners can participate in and jointly maintain this project.
If you are interested in this, please contact my Gmail address :wuzhihao7788@gmail.com. I look forward to working with you to improve and make progress.
