[简体中文](MODEL_ZOO_cn.md)|English

  

# Model Libraries And Bbaselines

  

## Test Environment

- Python 3.7

- PyTorch >=1.5

- CUDA 10.1

### YOLO

  

- **Important note:** Due to my limited resources, only a single 1080Ti video card can be used for training, and the training cycle is long after the complete training.To illustrate this framework is trainable, inferential, and testable.The pre-training model given in this paper is only the model of 24 epoch trained, and the loss value is still declining, which has not been fully trained, so it is only for your reference.Developers with the right conditions can complete this training, and I hope you can train the good pre-training model to provide, for everyone to use.I will also make a statement and thank you in an important place.

  

- Rendering

  
  

- YOLOv4：

<div  align="center">

<img  src="./images/test/bus-yolov4.jpg"  width=500  />

</div>

<div  align="center">

<img  src="./images/test/zidane-yolov4.jpg"  width=500  />

</div>

  

- YOLOv5-l:

<div  align="center">

<img  src="./images/test/bus-yolov5.jpg"  width=500  />

</div>

<div  align="center">

<img  src="./images/test/zidane-yolov5.jpg"  width=500  />

</div>

  

- PP-YOLO:

<div  align="center">

<img  src="./images/test/bus-ppyolo.jpg"  width=500  />

</div>

<div  align="center">

<img  src="./images/test/zidane-ppyolo.jpg"  width=500  />

</div>

| network | pre-training data set | input size | epoch | graphics card type | number of images per GPU |inference time (FPS) |cocotools AP<sup>val</sup>| AP<sub>50</sub> | baidu network disk | Google network disk | configuration file |logs | 
| :--------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: | :-------: | :-------: |  :-------: |    :-------: |     :-------: |  
| YOLOv4   | MSCOCO | 608  |24 |2070|2|  26.05ms|27.6|-|  [link](https://pan.baidu.com/s/1dTPRFnsMKKc2H0JalydY5Q) Extra code:yolo  | [link](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov4_coco_100e.py) |  [Statistical log](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l.log) |
| YOLOv5-l   | MSCOCO | 640|24 |  2070  |    2    |   19.1ms  |32.5|37.5|  [link](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) Extra code:yolo  | [link](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) | [Statistical log](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l.log) | 
| PP-YOLO   | MSCOCO | 608 |12 |  2070  |    4    |   270e |      -     |      -        |  [link](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) Extra code:yolo  | [link](https://drive.google.com/file/d/1jj21R1V9mHsbsxH-SZeq58FmariEgzQK/view?usp=sharing) | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/ppyolo_coco_100e.py) | [Statistical log](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l.log) | 
| YOLOv5-l+TTA   | MSCOCO | 640|24 |  2070  |    2    |   19.1ms  |33.6|38.3|  [link](https://pan.baidu.com/s/1m1DYtyEqCBmwK7Gq0BWB8w) Extra code:yolo  | [link](https://drive.google.com/file/d/16YcoQLlD7bYlQtHoBb1vf25Sx7lLNZFh/view?usp=sharing) | [config file](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/cfg/yolov5_coco_100e.py) | [Statistical log](https://github.com/wuzhihao7788/yolodet-pytorch/blob/master/docs/logs/yolov5-l-tta.log) | 
