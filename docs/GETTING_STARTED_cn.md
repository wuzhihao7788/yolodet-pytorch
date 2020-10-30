简体中文 | [English](GETTING_STARTED.md)
# 开始

这个页面提供了关于YOLODetection使用的基本教程。

有关安装说明，请参阅[INSTALL.md](INSTALL_cn.md)。


## 模型训练

YOLODetection提供执行单卡多卡的训练。
所有输出(日志文件和检查点)将保存到工作目录中。

这是由配置文件中的`work_dir`指定的。

**\*Important\***: 配置文件的默认学习率是1个gpu和小批次大小为2，累计到64个批次大小进行梯度更新。

根据[余弦败火规则](https://arxiv.org/abs/1706.02677)，如果你使用不同的GPU或每个GPU的图像，你需要设置与批大小成比例的学习率，配置文件中的`batch_size`和`subdivisions`确定。

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

### 用于测试图像的高级api接口

下面是一个构建模型和测试给定图像的示例。

```python
from yolodet.apis.inference import inference_detector,init_detector,show_result
import os

config_file ='cfg/yolov4_gpu.py'
checkpoint_file ='work_dirs/yolov4/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img ='test.jpg' # or img = mmcv.imread(img), which will only load it once
result,t = inference_detector(model, img)
print('%s Done , object num : %s .time:(%.3fs)' % (img,len(result), t))
# visualize the results in a new window
show_result(img, result, model.CLASSES)

```

