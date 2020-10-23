[简体中文](TRANSFORMS_cn.md) | English
## Training data preparation
This part mainly explains data loading, preprocessing, formatting, and data augmentation.
The entire process is executed sequentially in a pipeline manner, and each step of the operation is allowed to receive a dict table and output a dict table for the next conversion. Until all the pipelines are executed, the final dict table is returned.

The configuration items of YOLOv4 related to the pipeline are given below.
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
  dict(type='Resize',img_scale=img_scale,letterbox=True, auto=True,scaleup=True),
  dict(type='Normalize'),
  dict(type='ImageToTensor'),
  dict(type='Collect',keys=['img']),
]
  
val_pipeline = [
  dict(type='LoadImageFromFile',to_rgb=True),
  dict(type='Resize',img_scale=img_scale,letterbox=True, auto=False,scaleup=False),
  dict(type='Normalize'),
  dict(type='ImageToTensor'),
  dict(type='Collect'),
]
```
For each operation, list all related dict fields that can be added/updated/deleted.

### Description of main fields
- `Img`: When the data is loaded, it is the numpy 3D data of the original image RGB, and subsequent enhancements, formatting, etc. are the processed image numpy 3D data
- `Ori_shape`: the size of the original picture [h,w,d]
- `Img_shape`: The size of the converted picture [h,w,d]
- `gt_bboxes`: Truth box [x1y1x2y2], when the data is loaded, the data will be normalized, the range is [0,1], the normalization method is: x1 = x1/w, y1 = y1/h, x2 = x2/w, y2 = y2/h
- `gt_class`: index of category
- `gt_score`: score, the default is 1, if the MixUp method is used, the gt_score value may not be 1

### Data Loading
`LoadImageFromFile`
- Added: img, img_shape, ori_shape,gt_bboxes, gt_class, gt_score
- Description: It will load basic information such as pictures and tags from the configuration file

### Data preprocessing
`Mosaic`
- Update: img, img_shape,gt_bboxes, gt_class
- Description: The mosaic operation will randomly stitch four pictures into one picture

`Resize`
- Update: img, img_shape,gt_bboxes, gt_class
- `letterbox`: If it is False, it means proportional zooming. If it is True, the image is zoomed proportionally, the image is centered, and the two insufficient sides are filled with gray (114).
- `Auto`: This parameter is valid when letterbox=True. For the insufficient filling size of the two sides, auto is True and fills in the minimum multiple (32), and False fills in img_scale.
- Description: Resize, zoom pictures in equal proportions.

`RandomAffine`
- Update: img, img_shape,gt_bboxes, gt_class
- `degrees`: rotation angle.
- `translate`: Translation range.
- `scale`: zoom ratio.
- `shear`: Cut size.
- Description: RandomAffine random radiation transformation, mainly including rotation, scaling, shearing and translation.

`RandomHSV`
- Update: img
- Description: RandomHSV random color jitter, including hue, saturation and exposure.

`RandomFlip`
- Update: img,gt_bboxes
- Description: RandomFlip flips randomly.

### Format

`Normalize`
- Update: img
- Description: Normalize the picture is normalized, the range is [0,1]. The color value of each channel of RGB/255.

`ImageToTensor`
- Update: img
- Description: ImageToTensor converts the picture into a pytorch Tensor tensor, and changes it to the shape required by the model input Input [d,h,w]

`Collect`
- Add: img_meta (`meta_keys` key is specified by `meta_keys` parameter)
- Remove: Remove all `keys` except the specified `keys`
