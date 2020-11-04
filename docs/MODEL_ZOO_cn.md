简体中文| [English](MODEL_ZOO.md)

# 模型库和基线  
  
## 测试环境  
  
- Python 3.7  
- PyTorch >=1.5  
- CUDA 10.1  
   
 ### YOLO 
 
 - **重要说明:** 由于本人资源有限，只有单张1080Ti的显卡可供训练，完整训练完训练周期较长。为了说明本框架可训练，推断和测试。本次给出的预训练模型只是训练24个epoch的模型，损失值还在下降中，未训练充分，仅供大家参考。如有条件的开发者可完成本次训练，并希望你能把训练好的预训练模型提供出来，供大家使用。我也会在重要位置进行声明和感谢。


| 网络| 预训练数据集 | 输入尺寸 | epoch | 显卡类型 |推理时间(fps)|cocotools AP<sup>val</sup>| AP<sub>50</sub> | 百度网盘 | 谷歌网盘 | 配置文件 |日志| 
| :--------: | :--: | :-----: | :-----: |:------------: |:-------: | :----: | :-------: | :-------: |  :-------: |    :-------: |     :-------: |  
| YOLOv4   | MSCOCO | 608  |24 |2070|23ms|32.3|35.9|  [链接](https://pan.baidu.com/s/1InedBkJcR_j6E6vuDZnH-w) 提取码:yolo  | [链接](https://drive.google.com/file/d/1Yh0JY_vd8F1AUfTyK2Y69P2elX7LqGA-/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov4_coco_100e.py) |  [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov4.log) |
| YOLOv5-l   | MSCOCO | 640|24 |  2070  | 19ms  |32.5|37.5|  [链接](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) 提取码:yolo  | [链接](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) | [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l.log) | 
| PP-YOLO   | MSCOCO | 608 |24 |  2070  | 20ms |  44.5  |  49.4   |  [链接](https://pan.baidu.com/s/1Kso34r_um0iy4Emso5Jlqw) 提取码:yolo  | [链接](https://drive.google.com/file/d/1k0VEQjG9SU5eBsyy0n3eBciAW0byD7rD/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/ppyolo_coco_100e.py) | [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/ppyolo.log) | 
| YOLOv4-sam   | MSCOCO | 608  |33 |2070|22ms|34.6| 38.1|  [链接](https://pan.baidu.com/s/1InedBkJcR_j6E6vuDZnH-w) 提取码:yolo  | [链接](https://drive.google.com/file/d/1YgKoYfz7MVPRhANgWsNLRW3kMpnZY158/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov4_sam_coco_100e.py) |  [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov4-sam.log) |
| YOLOv5-l+TTA   | MSCOCO | 640|24 |  2070  | 52ms  |33.6|38.3|  [链接](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) 提取码:yolo  | [链接](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) | [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l-tta.log) | 
| YOLOv4+TTA   | MSCOCO | 640|24 |  2070  | 59ms  |32.9|36.2|  [链接](https://pan.baidu.com/s/1InedBkJcR_j6E6vuDZnH-w) 提取码:yolo  | [链接](https://drive.google.com/file/d/1Yh0JY_vd8F1AUfTyK2Y69P2elX7LqGA-/view?usp=sharing) | [配置文件](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) | [统计日志](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov4-tta.log) | 


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

 - YOLOv4-sam：
 <div align="center">
  <img src="./images/test/bus-yolov4-sam.jpg" width=500 />
</div>
<div align="center">
  <img src="./images/test/zidane-yolov4-sam.jpg" width=500 />
</div>

