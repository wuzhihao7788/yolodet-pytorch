#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 10:46
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: registry.py 
# @Software: PyCharm
# @github ：https://github.com/wuzhihao7788/yolodet-pytorch

               ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃              ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━-┓
                ┃Beast god bless┣┓
                ┃　Never BUG ！ ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
=================================================='''
#Register completed module

# transforms implementation list for data augmented , data transforms and data enhance
TRANSFORMS = dict(
    #从文件中加载图片，gt_class ,gt_bboxes等信息
    LoadImageFromFile='yolodet.dataset.pipelines.transforms.LoadImageFromFile',
    #随机噪声
    RandomNoise='yolodet.dataset.pipelines.transforms.RandomNoise',
    RandomBlur='yolodet.dataset.pipelines.transforms.RandomBlur',
    #随机颜色增强
    RandomHSV='yolodet.dataset.pipelines.transforms.RandomHSV',
    #随机镜像（翻转）
    RandomFlip='yolodet.dataset.pipelines.transforms.RandomFlip',
    MixUp='yolodet.dataset.pipelines.transforms.MixUp',
    Mosaic='yolodet.dataset.pipelines.transforms.Mosaic',
    Collect='yolodet.dataset.pipelines.transforms.Collect',
    Resize='yolodet.dataset.pipelines.transforms.Resize',
    #随机剪切
    RandomCrop='yolodet.dataset.pipelines.transforms.RandomCrop',
    ToTensor='yolodet.dataset.pipelines.transforms.ToTensor',
    Normalize='yolodet.dataset.pipelines.transforms.Normalize',
    ImageToTensor='yolodet.dataset.pipelines.transforms.ImageToTensor',
    #随机尺度（多尺度支持）
    RandomShape='yolodet.dataset.pipelines.transforms.RandomShape',
    #随机平移
    RandomTranslation='yolodet.dataset.pipelines.transforms.RandomTranslation',
    #随机放射变换（平移，缩放，旋转）
    RandomAffine='yolodet.dataset.pipelines.transforms.RandomAffine',
)

#dataset implementation
DATASET = dict(
    Custom='yolodet.dataset.custom.Custom'
)

# backbone net
BACKBONES = dict(
    CSPDarknet='yolodet.models.backbones.Darknet.CSPDarknet',
    ResNet='yolodet.models.backbones.resnet.ResNet',
    Res2Net='yolodet.models.backbones.res2net.Res2Net',
    YOLOv5Darknet='yolodet.models.backbones.yolov5.YOLOv5Darknet'
)

# detectors
DETECTORS = dict(
    YOLOv4Detector='yolodet.models.detectors.YOLOv4Detector.YOLOv4Detector',
    YOLOv5Detector='yolodet.models.detectors.YOLOv5Detector.YOLOv5Detector',
    PPYOLODetector='yolodet.models.detectors.PPYOLODetector.PPYOLODetector'
)
#necks
NECKS = dict(
    PANet='yolodet.models.necks.panet.PANet',
    YOLOv5FPN='yolodet.models.necks.yolov5.YOLOv5FPN',
    PPFPN='yolodet.models.necks.fpn.PPFPN'
)

HEADS = dict(
    YOLOv4Head='yolodet.models.heads.yolov4.YOLOv4Head',
    YOLOv5Head='yolodet.models.heads.yolov5.YOLOv5Head',
    PPHead='yolodet.models.heads.ppyolo.PPHead'
)

LOSS = dict(
    IOU_Loss='yolodet.models.loss.bbox_loss.IOU_Loss',
    Class_Loss='yolodet.models.loss.class_loss.Class_Loss',
    Conf_Loss='yolodet.models.loss.confidence_loss.Conf_Loss',
    IOU_Aware_Loss='yolodet.models.loss.bbox_loss.IOU_Aware_Loss',
    Focal_Loss='yolodet.models.loss.focalloss.FocalLoss',
)

HOOKS = dict(
    FixedLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.FixedLrUpdaterHook',
    StepLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.StepLrUpdaterHook',
    ExpLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.ExpLrUpdaterHook',
    PolyLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.PolyLrUpdaterHook',
    InvLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.InvLrUpdaterHook',
    CosineLrUpdaterHook='yolodet.models.hooks.lr_updater_hook.CosineLrUpdaterHook',
    OptimizerHook='yolodet.models.hooks.optinizer_hook.OptimizerHook',
    EvalHook='yolodet.models.hooks.eval_hook.EvalHook',
    DistEvalHook='yolodet.models.hooks.eval_hook.DistEvalHook',
    CheckpointHook='yolodet.models.hooks.checkpoint_hook.CheckpointHook',
    TextLoggerHook='yolodet.models.hooks.logger_hook.TextLoggerHook',
    TensorboardLoggerHook='yolodet.models.hooks.logger_hook.TensorboardLoggerHook',
    IterTimerHook='yolodet.models.hooks.iter_timer_hook.IterTimerHook',
)
