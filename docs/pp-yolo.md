[简体中文](pp-yolo_cn.md) | English

# PP-YOLO model

## Content
- [Introduction](#Introduction)
- [Network Structure and Features](#Network-Structure-and-Features)
- [Difference](#Difference)
- [How To Use](#How-To-Use)

## Introduction

[PP-YOLO](https://arxiv.org/abs/2007.12099) is a model of YOLOv3 optimized and improved by Baidu AI. The accuracy (COCO data set mAP) and reasoning speed given by Baidu are better than [YOLOv4] (https://arxiv.org/abs/2004.10934) model. [COCO](http://cocodataset.org) The accuracy of the test-dev2017 data set is 45.9%. The inference speed of FP32 is 72.9 FPS on the single card V100, and the inference speed of FP16 is 155.6 FPS when TensorRT is turned on on the V100.
## Network Structure and Features
#### Network structure
<div align="center">
  <img src="./images/pp-yolo.png"/>
</div>

#### PP-YOLO features:
- Backbone network: ResNet50vd-DCN
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- IOU_Aware_Loss
- Better pre-training model (knowledge distillation)
- Additional description of IOU_Aware_Loss loss function. In the original YOLO series, the confidence score is the classification probability multiplied by the target object score. The accuracy of the positioning of the detection frame is not considered. The loss value is used to measure the accuracy of the detection frame positioning.

## Difference
- The ResNet50 used by Baidu adopts the ResNet-D branch structure. The pre-training model I used is officially provided by pytorch, and the structure is the branch structure of ResNet-B. Baidu's ResNet50vd pre-training model uses knowledge distillation, and the accuracy is better than the official pytorch model.
- C5->P5(input:512,ouput:512), C4->P4(input:512,ouput:256), C3->P3(input:256,ouput:128) in the paper. In the actual code implementation, because the ResNet output is [512,1024,2048], the actual code implementation is C5->P5(input:2048,ouput:512), C4->P4(input:1024,ouput:256), C3->P3(input:512,ouput:128).
- The position of C5-->P5 DropBlock in the latest paddle code implementation is inconsistent with the position given in the paper. In the paper, Drop Block is used in the first Conv Block, but the actual code is to use Drop Block in the second Conv Block. By default, we are consistent with the paddle code implementation. You can modify the PFPPN configuration item of neck and set `second_drop_block=False` to keep it consistent with the original paper.
- IOU_Aware_Loss
- Regarding the definition of positive and negative samples, PP-YOLO adopts the definition method of YOLO's original series, and the definition of loss function is more xy_loss, wh_loss, IOU_Aware_Loss. However, actual experiment comparisons found that using the YOLOv5 positive and negative sample division definition and loss function definition can make the model converge faster, and exceed the original PP-YOLO series' division and loss definition of positive and negative samples. If the card resources are insufficient and you want to converge the model in a short time, you can use yolov5's positive and negative sample partition and loss function definition to converge the model. The default is to use the YOLOv5 method, the relevant parameters are `yolo_loss_type=yolov5`, for details, please refer to the YOLOv5 chapter description, set to None to use the loss function definition defined by PP-YOLO, and set to yolov4 to use the original YOLO series loss function.
- The model training method adopts the Mosaic method, instead of the official MixUP method, you can modify the relevant configuration items by yourself

## How To Use

### Preparation
Please download the ResNet50 pre-training model yourself
Download link: https://download.pytorch.org/models/resnet50-19c8e357.pth
Modify the configuration item `pretrained=ResNet50 local storage location`
```shell
type='PPYOLODetector',
pretrained='Local storage location of ResNet50 pre-training model',
backbone=dict(
  type='ResNet',
  depth=50,
  num_stages=4,
  out_indices=(1, 2, 3),
  frozen_stages=1,
  norm_cfg=dict(type='BN', requires_grad=True),
  norm_eval=True,
  style='pytorch',
  dcn=dict(type='DCNv2',deformable_groups=1, fallback_on_stride=False),
  # dcn=None,
  stage_with_dcn=(False, False, False, True)
)
```
Prepare the data set required for training by yourself, and specify the location of the data to be trained. For specific operations, please refer to 【[here](INSTALL.md)】 For data set preparation, please refer to the relevant chapters of yolov4, click 【[here](yolov4.md) 】Arrive quickly.


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
