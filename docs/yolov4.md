[简体中文](yolov4_cn.md) | English

# YOLOv4 model

## Content
- [Introduction](#Introduction)
- [Network Structure and Features](#Network-Structure-and-Features)
- [Training Skills](#Training-Skills)
- [How To Use](#How-To-Use)

## Introduction

[YOLOv4](https://arxiv.org/abs/2004.10934) has not changed much from YOLOv3 at the network structure level. It's just that the backbone network replaced the original Darknet53 by CSPDarknet53, while introducing SPP and PAN ideas for feature fusion of different scales.

- Proposed an efficient and powerful target detection model. It allows everyone to use 1080 Ti or 2080 Ti GPU to train ultra-fast and accurate target detectors.
- Verified many SOTA deep learning target detection training techniques in recent years.
- Modified many SOTA methods to make them more efficient for single GPU training, such as CBN, PAN, SAM, etc.

## Network Structure and Features
#### Network structure
<div align="center">
  <img src="./images/yolov4.png"/>
</div>

#### Network details

Use [Netron](https://github.com/lutzroeder/Netron) to visualize the YOLOv4 model structure. Click【[here](./images/yolov4-detail.png)】 for details


#### YOLOv4 features:
- Backbone network: CSPDarknet53
- [Mish activation](https://arxiv.org/abs/1908.08681)
- [FPN+PAN structure](https://arxiv.org/abs/1803.01534)
- [DIoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- Mosaic
- Label Smooth
- SAM(Spartial Attention Module)

## Training Skills
- [Exponential Moving Average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- Warm up
- Gradient shear
- Gradient cumulative update
- Multi-scale training
- Learning rate adjustment: Consine
- Label Smooth
- Regarding the definition of positive and negative samples, in actual experiment comparison, it is found that the use of YOLOv5 positive and negative sample definition and loss function definition can make the model converge faster, exceeding the original YOLO series' definition of positive and negative samples and loss. If the card resources are insufficient and you want to converge the model in a short time, you can use yolov5's positive and negative sample partition and loss function definition to converge the model. The default is to use the YOLOv5 method, and the relevant parameter is `yolo_loss_type=yolov5`. For details, please refer to the YOLOv5 chapter. Set to None or yolov4 to use the original YOLO series loss function.

## How To Use

### Preparation
Prepare the data set required for training by yourself, and specify the location of the data to be trained. For specific operations, please refer to 【[here](INSTALL.md)】for data set preparation.
Find the yolov4 configuration file and modify the following parts
```python
data = dict(
  batch_size=64, #cumulative batch size, the framework will cumulatively update the gradient after the batch size is accumulated
  subdivisions=8, #The number of batches, the actual batch size is batch_size/subdivisions, modify the value size according to your own machine configuration. If the video memory is sufficient, the value can be set smaller, if the video memory is insufficient, the value can be set larger.
  workers_per_gpu=2, #dataloader The number of working threads for loading data
  train=dict(
        type='Custom',
  data_root=r'xxxx', #training data root directory
  ann_file=r'annotations/train.txt', #Tag file location, relative path. data_root+ann_file
  img_prefix='images', #Picture storage path directory, relative path. data_root+img_prefix
  name_file='annotations/label.names', #label index corresponding name
  pipeline=train_pipeline
    ),
  val=dict(
        type='Custom',
  data_root=r'xxxx', # Same as above
  ann_file=r'annotations/val.txt', # Same as above
  img_prefix='images', #same as above
  name_file='annotations/label.names', #same as above
  pipeline=val_pipeline
    ),
  test=dict(
        type='Custom',
  data_root=r'xxx', # Same as above
  ann_file=r'annotations/test.txt', # Same as above
  img_prefix='images', #same as above
  name_file='annotations/label.names', #same as above
  pipeline=test_pipeline
    )
)
```


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
