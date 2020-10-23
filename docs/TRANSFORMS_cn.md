简体中文 | [English](TRANSFORMS.md)
## 训练数据准备
此部分内容主要对数据加载、预处理、格式化和数据增广等进行说明。
整个过程采用一种管道的方式顺序执行，每步操作都允许接收一个dict表，并为下一个转换输出一个dict表。直到所有的管道执行完成，返回最终的dict表。

下面给出YOLOv4有关管道的配置项。
```python
img_scale = 608  
train_pipeline = [  
    dict(  
        type='Mosaic',  
		img_scale=img_scale,  
        transforms=[  
	        dict(type='LoadImageFromFile',to_rgb=True),   
			dict(type='Resize',img_scale=img_scale,letterbox=False),   
			]),
	dict(type='RandomAffine',degrees=0, translate=0, scale=.5, shear=0.0),  
	dict(type='RandomHSV'),  
	dict(type='RandomFlip'),  
	dict(type='Normalize'),  
	dict(type='ImageToTensor'),  
	dict(type='Collect'),  
]  
test_pipeline = [  
  dict(type='LoadImageFromFile',to_rgb=True),  
  dict(type='Resize',img_scale=img_scale,letterbox=True, auto= True,scaleup=True),  
  dict(type='Normalize'),  
  dict(type='ImageToTensor'),  
  dict(type='Collect',keys=['img']),  
]  
  
val_pipeline = [  
  dict(type='LoadImageFromFile',to_rgb=True),  
  dict(type='Resize',img_scale=img_scale,letterbox=True, auto= False,scaleup=False),  
  dict(type='Normalize'),  
  dict(type='ImageToTensor'),  
  dict(type='Collect'),  
]
```
对于每个操作，列出全部可添加/更新/删除的相关dict字段。

### 主要字段说明
- `img`:数据加载时为原始图片RGB的numpy三维数据，后续有增强，格式化等为处理后的图片numpy三维数据
- `ori_shape`:原始图片的大小[h,w,d]
- `img_shape`:转换后的图片的大小[h,w,d]
- `gt_bboxes`:真值框[x1y1x2y2]，数据加载时，会对数据进行归一化,范围为[0,1]，归一化方式为：x1 = x1/w,  y1= y1/h, x2 = x2/w, y2 = y2/h
- `gt_class`:类别的索引
- `gt_score`:评分，默认为1，如果采用MixUp方式，gt_score值不一定为1

### 数据加载
`LoadImageFromFile`
- 增加: img, img_shape, ori_shape,gt_bboxes, gt_class, gt_score
- 说明:会从配置文件中加载图片，标签等基本信息

### 数据预处理
`Mosaic`
- 更新: img, img_shape,gt_bboxes, gt_class
- 说明:mosaic操作会随机拼接四张图片为一张图片

`Resize`
- 更新: img, img_shape,gt_bboxes, gt_class
- `letterbox`:如果为False 代表等比例缩放，如果为True，图片等比例缩放，图片居中，不足的两条边用灰色（114）填充。
-  `auto`:此参数在letterbox=True时有效，对于不足的两条边填充大小，auto为True按最小倍数（32）填充够，为False按img_scale填充完整。
- 说明:Resize，等比例缩放图片。

`RandomAffine`
- 更新: img, img_shape,gt_bboxes, gt_class
- `degrees`:旋转角度。
-  `translate`:平移范围。
- `scale`:缩放比例。
- `shear`:剪切大小。
- 说明:RandomAffine 随机放射变换，主要包含旋转,缩放,剪切和平移等。

`RandomHSV`
- 更新: img
- 说明:RandomHSV 随机颜色抖动，包含色调，饱和度和曝光度等。

`RandomFlip`
- 更新: img,gt_bboxes
- 说明:RandomFlip 随机翻转。

### 格式化

`Normalize`
- 更新: img
- 说明:Normalize 图片归一化，范围[0,1]。 RGB各通道颜色值/255。

`ImageToTensor`
- 更新: img
- 说明:ImageToTensor 将图片转换为pytorch Tensor张量，改变为模型输入Input需要的形状[d,h,w]

`Collect`
- 添加:img_meta (`meta_keys`键由`meta_keys`参数指定)
- 移除: 移除所有除指定的`keys`外所有其他`keys`









