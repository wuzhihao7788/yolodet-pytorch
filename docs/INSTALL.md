[简体中文](INSTALL_cn.md) | English
## Installation

### Claim

- Linux (current code has not been tested in Windows environment)

- python3.7 + (python2 is not supported)

- PyTorch 1.5 or higher

- CUDA 10.0 or higher

- NCCL 2

- GCC(G++) 4.9 or above

The following versions of operating system and software have been tested:

- Operating system: Ubuntu 16.04/18.04

- CUDA: 9.2/10.0

- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2

- GCC (G + +): 4.9/5.3/7.3

### Install YOLODetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n yolodet python=3.7 -y
conda activate yolodet
```

b. Follow [official instructions](https://pytorch.org/) to install PyTorch stable or nightly and torchvision, for example:

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the yolodet-pytorch library.

```shell
git clone https://github.com/wuzhihao7788/yolodet-pytorch.git
cd yolodet-pytorch
```

d. Install yolodet (other dependencies will be installed automatically).
- 1. If you need to install, compile and install when using DCN, you can install it in the following way
```shell
python setup.py develop # or "pip install -v -e ."
```
- 2. If you do not use DCN, you can install dependencies in the following ways
```shell
pip install -r requirements.txt
```

### Prepare the dataset

It is recommended to connect the root directory of the data set to `$YOLODET/data`.

If your folder structure is different, you may need to change the corresponding path in the configuration file.


```
yolodet-pytorch
├── yolodet
├── tools
├── cfg
├── data
│ ├── your data root    #Your data set root directory
│ │ ├── annotations     #label storage location
│ │ │ ├── train.txt     #Training data set label file. Data format: [picture name x1,y1,x2,y2,label] For example: 5979.jpg 253,420,406,744,0 25,40,46,44,1
│ │ │ ├── val.txt       #Verify the data set label file. The data format is the same as above
│ │ │ ├── test.txt      #Test data set label file. The data format is the same as above
│ │ ├── images #Picture storage location
│ │ ├── label.names     #label name storage location, press label index, store by row
```
