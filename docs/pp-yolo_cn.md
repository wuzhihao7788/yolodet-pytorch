简体中文 | [English](pp-yolo.md)

# PP-YOLO 模型

## 内容
- [简介](#简介)
- [网络结构和特点](#网络结构和特点)
- [不同点](#不同点)
- [如何使用](#如何使用)

## 简介

[PP-YOLO](https://arxiv.org/abs/2007.12099)是百度AI优化和改进的YOLOv3的模型，百度官方给出的精度(COCO数据集mAP)和推理速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)模型。[COCO](http://cocodataset.org) test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。
## 网络结构和特点
#### 网络结构
<div align="center">
  <img src="./images/pp-yolo.png" />
</div>

#### PP-YOLO特点：
- 骨干网络: ResNet50vd-DCN
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- IOU_Aware_Loss
- 更优的预训练模型(知识蒸馏)
- 额外对IOU_Aware_Loss损失函数说明，原YOLO系列中，置信度得分为分类概率乘以目标物体得分。未考虑检测框定位的准确度。该损失值用于衡量检测框定位的准确度。

## 不同点
- 百度使用的ResNet50采用的是ResNet-D分支结构。而我使用的预训练模型为pytorch官方提供的，结构为ResNet-B的分支结构。百度的ResNet50vd的预训练模型采用了知识蒸馏，精度要好于pytorch官方模型。
- 论文中C5->P5(input:512,ouput:512)，C4->P4(input:512,ouput:256)，C3->P3(input:256,ouput:128)  。而实际代码实现中因为ResNet输出为[512,1024,2048]，所以实际代码实现为C5->P5(input:2048,ouput:512) ，C4->P4(input:1024,ouput:256)  ，C3->P3(input:512,ouput:128)。  
- 最新的paddle代码实现对于C5-->P5 DropBlock的位置和论文中给出的位置不一致。论文中是在第一个Conv Block中采用Drop Block，但实际代码是在第二个Conv Block块中采用Drop Block。默认我们是与paddle代码实现保持一致。可以修改neck的PPFPN配置项，设置`second_drop_block=False`，即可和原论文保持一致。
- IOU_Aware_Loss
- 有关正负样本划分定义，PP-YOLO采用的是YOLO原系列的定义方式，损失函数的定义多了xy_loss，wh_loss，IOU_Aware_Loss的损失。但是实际实验对比发现采用YOLOv5正负样本划分定义和损失函数定义，能使模型收敛速度更快，超过原PP-YOLO系列对正负样本的划分和损失定义。对于如果卡资源不充足，想在短时间内收敛模型，可采用yolov5的正负样本划分和损失函数定义，收敛模型。默认是采用YOLOv5的方式，相关参数为`yolo_loss_type=yolov5`，具体细节可参考YOLOv5章节说明，设置为None将采用PP-YOLO定义的损失函数定义，设置为yolov4将采用原始YOLO系列损失函数。
- 模型训练方法采用Mosaic方式，未采用官方的MixUP方式，可自行修改相关配置项

## 如何使用

### 准备
请自行下载ResNet50预训练模型
下载地址：https://download.pytorch.org/models/resnet50-19c8e357.pth
修改配置项`pretrained=ResNet50本地存放位置`
```shell
type='PPYOLODetector',  
pretrained='ResNet50预训练模型本地存放位置',  
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
自行准备训练需要的数据集，指定需要训练的数据位置，具体操作请查看【[这里](INSTALL_cn.md)】有关数据集准备，可查看yolov4相关章节，点击【[这里](yolov4_cn.md)】快速到达。


### 使用GPU训练
```shell
python tools/train.py ${CONFIG_FILE}
```
如果您想在命令中指定工作目录，可以添加一个参数`--work_dir ${YOUR_WORK_DIR}`。
例如采用YOLOv4训练模型:
```shell
python tools/train.py cfg/yolov4_coco_100e.py --device ${device} --validate
```

### 使用指定gpu训练

```shell
python tools/train.py ${CONFIG_FILE} --device ${device} [optional arguments]
```
例如采用YOLOv4训练模型:
```shell
python tools/train.py cfg/yolov4_coco_100e.py --device 0,1,2 --validate
```

可选参数:

- `--validate`(**强烈建议**):在训练epoch期间每一次k(默认值是1，可以像这样修改[this](../cfg/yolov4_coco_gpu.py#L138))来执行评估。

- `--work_dir ${WORK_DIR}`:覆盖配置文件中指定的工作目录。
- `--device ${device}`: 指定device训练, 0 or 0,1,2,3 or cpu，默认全部使用。

- `--resume_from ${CHECKPOINT_FILE}`:从以前训练的checkpoints文件恢复训练。
- `--multi-scale`:多尺度缩放，尺寸范围为训练图片尺寸+/- 50%

`resume_from` 和`load_from`的不同:

`resume_from`加载模型权重和优化器状态，并且训练也从指定的检查点继续训练。它通常用于恢复意外中断的训练。
`load_from`只加载模型权重，并且训练从epoch 0开始。它通常用于微调。



