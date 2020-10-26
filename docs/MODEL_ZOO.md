[简体中文](MODEL_ZOO_cn.md)|English

# Model Libraries And Bbaselines

## Test Environment
- Python 3.7  
- PyTorch >=1.5  
- CUDA 10.1  
   
 ### YOLO 

- **Important note:** Due to my limited resources, only a single 1080Ti video card can be used for training, and the training cycle is long after the complete training.To illustrate this framework is trainable, inferential, and testable.The pre-training model given in this paper is only the model of 12 epoch trained, and the loss value is still declining, which has not been fully trained, so it is only for your reference.Developers with the right conditions can complete this training, and I hope you can train the good pre-training model to provide, for everyone to use.I will also make a statement and thank you in an important place.

- Rendering


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


| network | pre-training data set | input size | epoch | graphics card type | number of images per GPU | inference time (FPS)| Box AP | baidu network disk |Google network disk |configuration file |
| :--------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: | :-------: | :-------: |  :-------: |  
| YOLOv4   | MSCOCO | 608  |12 |2070|2|  32.05ms|27.6|  0.276  | - | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov4_coco_100e.py) |  
| YOLOv5-l   | MSCOCO | 640|12 |  2070  |    2    |   270e  |      -        |  31.0  | - | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) |  
| PP-YOLO   | MSCOCO | 608 |12 |  2070  |    4    |   270e  |      -        |  28.2  | - | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/ppyolo_coco_100e.py) |  
