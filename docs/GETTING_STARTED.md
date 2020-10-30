[简体中文](GETTING_STARTED_cn.md) | English
# How To Start

This page provides a basic tutorial on the use of YOLODetection.

For installation instructions, see [INSTALL.md](INSTALL.md).


## Model training

YOLODetection provides training to perform single or multiple cards.
All output (log files and checkpoints) will be saved to the working directory.

This is specified by `work_dir` in the configuration file.

**Important**: The default learning rate of the configuration file is 1 gpu and the small batch size is 2, accumulating to 64 batch sizes for gradient update.

According to [Cosine Fighting Rules](https://arxiv.org/abs/1706.02677), if you use different GPUs or images of each GPU, you need to set a learning rate proportional to the batch size. `batch_size` and `subdivisions` are determined.

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

### Advanced api interface for testing images

Below is an example of building a model and testing a given image.

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
