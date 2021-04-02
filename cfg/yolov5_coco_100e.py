# YOLO model
model = dict(
    type='YOLOv5Detector',
    pretrained=None,
    depth_multiple = 1.33,  # model depth multiple 类似efficient net 代表深度   l:1 m:0.67 s:0.33 x:1.33
    width_multiple = 1.25,  # layer channel multiple 类似efficient net 代表宽度 卷积的输出通道数 l:1 m:0.75 s:0.50 x:1.25
    backbone=dict(
        type='YOLOv5Darknet',
        focus = [64,3,1],
        in_channels= 3,
        bottle_depths=[3, 9, 9, 3],
        shortcut = [True,True,True,False],
        out_channels = [128,256,512,1024],
        spp=[5, 9, 13],#spp 核大小
        out_indices=(2, 3, 4,),
        norm_type='BN',#支持BN，GN
        num_groups=None#norm_type=GN 需要给出组大小，通道整数倍
    ),
    neck=dict(
        type='YOLOv5FPN',
        in_channels = 1024,
        upsampling_mode = 'nearest',#``'nearest'``,``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.Default: ``'nearest'``
        out_channels = [512,256,256,512],
        bottle_depths=[3, 3, 3, 3],
        shortcut = [False,False,False,False],
        norm_type='BN',#支持BN，GN
        num_groups=None#norm_type=GN 需要给出组大小，通道整数倍
    ),
    head=dict(
        type='YOLOv5Head',
        in_channels=[256,512,1024],
        num_classes = 80, # class num 依据你自己的训练标签类别自行修改数量
        label_smooth=True, #是否启用label_smooth
        deta=0.01, #label_smooth的deta值
        anchors=[[10,13], [16,30], [33,23],[30,61], [62,45], [59,119],[116,90], [156,198], [373,326]],#锚框大小
        anchor_masks=[[0, 1, 2], [3, 4, 5],[6, 7, 8]],#锚框大小
        downsample_ratios=[8, 16, 32],
        nms_type='nms',#Support types :nms or soft_nms
        nms_thr=.5,#nms 阈值
        yolo_loss_type='yolov5',#损失函数类型支持，支持字段 yolov4，None
        ignore_thre=.4, #yolo原系列 置信度损失 负样本 neg_loss忽略部分的阈值
        conf_balances=[4, 1, 0.4], #对不同尺度的置信度权重设置[小，中，大]
        bbox_loss=dict(type='IOU_Loss', iou_type='GIOU',loss_weight=0.5),#IOU_loss 支持 IOU CIOU DIOU GIOU
        confidence_loss=dict(type='Conf_Loss',loss_weight=1,pos_weight=1.0),#confidence_loss
        class_loss=dict(type='Class_Loss',loss_weight=0.5,pos_weight=1.0),#class_loss
        norm_type='BN',#支持BN，GN
        num_groups=None#norm_type=GN 需要给出组大小，通道整数倍
    )
)

img_scale = 640 #图片缩放尺度
train_pipeline = [
    dict(
        type='Mosaic',#Mosaic
        img_scale=img_scale,
        transforms=[
    dict(type='LoadImageFromFile'),#从配置项加载图片和标签，真框等并对框进行归一化
    dict(type='Resize',img_scale=img_scale,letterbox=False),#Resize图片，此处letterbox必须为False
        ]),
    dict(type='RandomAffine',degrees=0, translate=0, scale=.5, shear=0.0),#随机放射变换
    dict(type='RandomHSV'),#随机颜色抖动
    dict(type='RandomFlip'),#随机翻转
    dict(type='Normalize'),#格式化数据主要对输入图片格式化，归一化图片 /255
    dict(type='ImageToTensor'), #将图片numpy转换为torch 张量，并reshape为模型Input shape
    dict(type='Collect'),#过滤不需要的字典数据
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',img_scale=img_scale,letterbox=True, auto= True,scaleup=True),#图片resize到img_scale，并最小自适应宽高
    dict(type='Normalize'),
    dict(type='ImageToTensor'),
    dict(type='Collect',keys=['img']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',img_scale=img_scale,letterbox=True, auto= False,scaleup=False),#图片resize到img_scale，缺失部分用灰色填充
    dict(type='Normalize'),
    dict(type='ImageToTensor'),
    dict(type='Collect'),
]
#具体配置请查看先关md文件说明，不在细说
data = dict(
    batch_size=64,
    subdivisions=16,
    workers_per_gpu=1,
    train=dict(
        type='Custom',
        data_root=r'data/mscoco',
        ann_file=r'annotations/train.txt',
        img_prefix='train2017',
        name_file='annotations/label.names',
        pipeline=train_pipeline
    ),
    val=dict(
        type='Custom',
        data_root=r'data/mscoco',
        ann_file=r'annotations/val.txt',
        img_prefix='val2017',
        name_file='annotations/label.names',
        pipeline=val_pipeline
    ),
    test=dict(
        type='Custom',
        data_root=r'data/mscoco',
        ann_file=r'annotations/test.txt',
        img_prefix='test2017',
        name_file='annotations/label.names',
        pipeline=test_pipeline
    )
)
#Model Exponential Moving Average https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
ema = dict(decay=0.9999, updates=0) #Exponential Moving Average 配置
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.937, weight_decay= 5e-4) #优化器 默认SGD
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2),detect_anomaly=False)#优化器配置项 grad_clip:梯度剪切，防止梯度爆炸或消失 detect_anomaly :该参数在debug模式下打开，可帮助调试代码，默认关闭，影响模型训练速度。
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=15000,
    # warmup_by_epoch=True,
    warmup_ratio=1.0 / 3,
    target_lr=0.0002,
    # step=[8,12,18,22])
    )
checkpoint_config = dict(interval=1) #模型保存策略，默认每个epoch都保存
#日志配置
log_config = dict(
    interval=2,#多少个批次打印一条日志
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
	
evaluation = dict(interval=2,save_best=True)#验证集配置，根据统计指标计算，默认给保存最好的模型
total_epochs = 100 #训练epoch大小
work_dir = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov5_hand' #模型保存文件目录，包含日志文件
load_from = None #用于加载已训练完模型，用于用较低学习率微调网络
resume_from = None #用于程序以外中断，继续训练
workflow = [('train', 1)]# 添加('val', 1)，训练集和验证集一块训练


log_info = dict(
    # 日志保留路径,默认保留在项目跟目录下的logs文件
    log_dir='',
    # 日志级别,默认INFO级别,ERROR,WARNING,WARN,INFO,DEBUG
    log_level='INFO',
    # 每天生成几个日志文件,默认每天生成1个
    log_interval=1,
    # #日志保留多少天,默认保留7天的日志
    log_backupCount=7,
)
