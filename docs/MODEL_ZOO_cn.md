简体中文| [English](MODEL_ZOO.md)

# 模型库和基线  
  
## 测试环境  
  
- Python 3.7  
- PyTorch >=1.5  
- CUDA 10.1  
   
 ### YOLO 
 
 - **重要说明:** 由于本人资源有限，只有单张1080Ti的显卡可供训练，完整训练完训练周期较长。为了说明本框架可训练，推断和测试。本次给出的预训练模型只是训练24个epoch的模型，损失值还在下降中，未训练充分，仅供大家参考。如有条件的开发者可完成本次训练，并希望你能把训练好的预训练模型提供出来，供大家使用。我也会在重要位置进行声明和感谢。
 - 效果图
 
 - YOLOv4：
 <div align="center">
  <img src="./images/test/bus-yolov4.jpg" width=500 />
</div>
<div align="center">
  <img src="./images/test/zidane-yolov4.jpg" width=500 />
</div>

- YOLOv5-l:
 <div align="center">
  <img src="./images/test/bus-yolov5.jpg" width=500 />
</div>
<div align="center">
  <img src="./images/test/zidane-yolov5.jpg" width=500 />
</div>

- PP-YOLO:
 <div align="center">
  <img src="./images/test/bus-ppyolo.jpg" width=500 />
</div>
<div align="center">
  <img src="./images/test/zidane-ppyolo.jpg" width=500 />
</div>

| 网络| 预训练数据集 | 输入尺寸 | epoch | 显卡类型 | 每张GPU图片个数 |推理时间(fps)|cocotools AP<sup>val</sup>| AP<sub>50</sub> | 百度网盘 | 谷歌网盘 | 配置文件 |  
| :--------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: | :-------: | :-------: |  :-------: |  
| YOLOv4   | MSCOCO | 608  |24 |2070|2|  26.05ms|27.6|-|  [链接](https://pan.baidu.com/s/1dTPRFnsMKKc2H0JalydY5Q) 提取码:yolo  | [链接](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov4_coco_100e.py) |  
| YOLOv5-l   | MSCOCO | 640|24 |  2070  |    2    |   19.1ms  |32.5|37.5|  [链接](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) 提取码:yolo  | [链接](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) |  
| PP-YOLO   | MSCOCO | 608 |12 |  2070  |    4    |   270e  |      -        |  [链接](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) 提取码:yolo  | [链接](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/ppyolo_coco_100e.py) |  
| YOLOv5-l+TTA   | MSCOCO | 640|24 |  2070  |    2    |   19.1ms  |33.6|38.3|  [链接](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) 提取码:yolo  | [链接](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) |  
