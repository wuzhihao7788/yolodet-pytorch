[简体中文](yolov5_cn.md) | English

# YOLOv5 model

## Content
- [Introduction](#Introduction)
- [Network Structure and Features](#Network-Structure-and-Features)
- [Training Skills](#Training-Skills)
- [How To Use](#How-To-Use)

## Introduction

Yolov5 is very similar in structure to YOLOv4. The difference is that its configuration is more flexible, and the depth and breadth of the network can be easily configured. A total of 4 versions of the network are given, namely **Yolov5s, Yolov5m, Yolov5l, Yolov5x** The structure of each model is different in the depth (number of blocks) and width (number of channels) of the BottleneckCSP module, which can meet the configuration model of the machine. This design structure is very similar to Google’s EfficientNet, and the depth of the network is also considered. The impact of breadth and resolution on the model.
## Network Structure and Features
#### Network structure
<div align="center">
  <img src="./images/yolov5.png"  />
</div>

#### Network details

Use [Netron](https://github.com/lutzroeder/Netron) to visualize the yolov5x model structure. Click【[here](./images/yolov5x-detail.png)】 for details


#### YOLOv5 features and skills:
- Backbone network: Focus structure, CSP structure
- [Mish activation](https://arxiv.org/abs/1908.08681)
- [FPN+PAN structure](https://arxiv.org/abs/1803.01534)
- [GIOU_Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- Mosaic
- Support adaptive anchor frame calculation
- Inferred support for adaptive image scaling
- Label Smooth
- Focal Loss
- Custom positive sample

## Training Skills
- [Exponential Moving Average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- Warm up
- Gradient shear
- Gradient cumulative update
- Multi-scale training
- Learning rate adjustment: Consine
- Label Smooth
- YOLOv5's definition of positive samples: as long as the ratio of the true frame to the given anchor frame is within 4 times under different scale anchor frames, the anchor frame can be responsible for predicting the true value frame. And according to the offset of gx and gy in the center of the grid, two additional grid coordinates will be added to predict. Through this series of operations, the number of positive samples is increased and the model convergence speed is accelerated. For the original YOLO series, for the true frame, only the anchor frame with the largest IOU intersection at this scale is responsible for predicting the true frame at different scales, and other anchor frames are not responsible. Therefore, due to the smaller positive sample size, the model converges faster. slow.

## How To Use

### Preparation

Prepare the data set required for training by yourself, and specify the location of the data to be trained. For specific operations, please see 【[here](INSTALL.md)】. For data set preparation, you can check the relevant chapters of yolov4, click 【[here](yolov4.md)】 to get there quickly.


### Use GPU training
```shell
python tools/train.py ${CONFIG_FILE}
```
If you want to specify the working directory in the command, you can add a parameter `--work_dir ${YOUR_WORK_DIR}`.

### Use assign GPU training

```shell
python tools/train.py ${CONFIG_FILE} --device ${device} [optional arguments]
```

Optional parameters:

- `--validate` (**strongly recommended**): every time k during the training epoch (the default value is 1, you can modify [this](../cfg/yolov4_coco_gpu.py#L138)) to execute Evaluation.

- `--work_dir ${WORK_DIR}`: Overwrite the working directory specified in the configuration file.
- `--device ${device}`: assign cuda device , 0 or 0,1,2,3 or cpu，use all by default。
- `--resume_from ${CHECKPOINT_FILE}`: Resume training from the checkpoints file of previous training.
- `--multi-scale`: Multi-scale scaling, the size range is +/- 50% of the training image size

The difference between `resume_from` and `load_from`:

`resume_from` loads the model weight and optimizer state, and the training continues from the specified checkpoint. It is usually used to resume training that was interrupted unexpectedly.
`load_from` only loads model weights, and training starts from epoch 0. It is usually used for fine-tuning.
