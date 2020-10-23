简体中文 | [English](INSTALL.md)
## 安装

### 要求

- Linux(目前代码未在Windows环境试验)

- python3.7 +(不支持python2)

- PyTorch 1.5或更高版本

- CUDA 10.0或更高

- NCCL 2

- GCC(G++) 4.9或以上

已经测试了以下版本的操作系统和软件:

-操作系统:Ubuntu 16.04/18.04

- CUDA: 9.2/10.0

- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2

- GCC (G + +): 4.9/5.3/7.3

### 安装YOLODetection

a.创建一个conda虚拟环境并激活它。

```shell
conda create -n yolodet python=3.7 -y
conda activate yolodet
```

b.按照[官方说明](https://pytorch.org/)安装PyTorch stable或nightly和torchvision，例如:

```shell
conda install pytorch torchvision -c pytorch
```

c.克隆yolodet-pytorch库。

```shell
git clone https://github.com/wuzhihao7788/yolodet-pytorch.git
cd yolodet-pytorch
```

d.安装yolodet(其他依赖项将自动安装)。
- 1.如果使用DCN需要安装编译安装，可通过如下方式安装
```shell
python setup.py develop  # or "pip install -v -e ."
```
- 2.如果不使用DCN，可通过以下方式安装依赖
```shell
pip install -r requirements.txt
```

### 准备数据集

建议将数据集的根目录软连接到`$YOLODET/data`。

如果你的文件夹结构不同，你可能需要改变配置文件中相应的路径。


```
yolodet-pytorch
├── yolodet
├── tools
├── cfg
├── data
│   ├── your data root     #你的数据集根目录
│   │   ├── annotations    #标签存放位置
│   │   │   ├── train.txt  #训练数据集标签文件。数据格式：[图片名称 x1,y1,x2,y2,label] 例如：59679.jpg 253,420,406,744,0 25,40,46,44,1
│   │   │   ├── val.txt    #验证数据集标签文件。数据格式同上
│   │   │   ├── test.txt   #测试数据集标签文件。数据格式同上
│   │   ├── images        #图片存放位置
│   │   ├── label.names   #标签名称存放位置,按标签索引，按行存储
```
