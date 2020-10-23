#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/27 18:39
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: transforms.py 
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
import math
import random
from collections import Sequence

import torch
import cv2
import numpy as np

import os.path as osp

from yolodet.dataset.pipelines.compose import Compose


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale

class LoadImageFromFile(object):

    def __init__(self,to_rgb=True):
        self.to_rgb = to_rgb
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, results):
        img_name = results['img_name']
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], img_name)
        else:
            filename = img_name

        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        y_true = results['y_true']
        num_bbox = len(y_true)

        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32) # gt_bbox[x1,y1,x2,y2]
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32) # gt_class:[number]
        gt_score = np.ones((num_bbox, 1), dtype=np.float32)  # gt_score ：default 1

        for i,yt in enumerate(y_true):
            gt_class[i][0] = yt[4]
            gt_bbox[i, :] = [yt[0]/w,yt[1]/h,yt[2]/w,yt[3]/h]#格式化坐标

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['gt_bboxes'] = gt_bbox
        results['gt_class'] = gt_class
        results['gt_score'] = gt_score

        return results


class RandomNoise(object):
    def __init__(self,gaussian_noise=50,rand_thr=0.5):
        self.rand_thr = rand_thr
        self.gaussian_noise = gaussian_noise

    def __call__(self, results):
        if random.random()> self.rand_thr:
            img = results['img']
            gaussian_noise = min(self.gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            gaussian_noise = random.randint(0, gaussian_noise)
            noise = np.random.normal(0, gaussian_noise, img.shape)
            img = img + noise
            results['img'] = img
            results['img_shape'] = img.shape

        return results

class RandomBlur(object):
    def __init__(self,rand_thr=0.5):
        self.rand_thr = rand_thr

    def __call__(self, results):
        if random.random()> self.rand_thr:
            img = results['img']
            dst = cv2.GaussianBlur(img, (3, 3), 0)
            results['img'] = dst
            results['img_shape'] = dst.shape
        return results

class RandomHSV(object):
    def __init__(self,hue=0.1,saturation=1.5,exposure=1.5,rand_thr=0.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.rand_thr = rand_thr


    def __call__(self, results):
        if random.random() < self.rand_thr:
            return results
        dhue = rand_uniform_strong(-self.hue,self.hue)  # 色调
        dsat = rand_scale(self.saturation)  # 饱和度
        dexp = rand_scale(self.exposure)  # 曝光度
        if dsat != 1 or dexp != 1 or dhue != 0:
            img = results['img']
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                img *= dexp

        results['img'] = img

        return results

class ImageToTensor(object):
    def __init__(self, keys=['img']):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results


class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        # results['img'] = imnormalize(results['img'], self.mean, self.std,
        #                                   self.to_rgb)
        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        im = results['img']
        # results['ori_img'] = results['img']
        im = im.astype(np.float32, copy=False)
        results['img'] = im/255

        # if 'gt_bboxes' not in results:
        #     return results

        # gt_bbox = results['gt_bboxes']

        # if 'num_bbox' in results:
        #     NUM_BOXES = results['num_bbox']
        #     if len(gt_bbox)!= NUM_BOXES:
        #         gt_class = results['gt_class']
        #         gt_score = results['gt_score']
        #         out_bboxes = np.zeros([NUM_BOXES, 4])
        #         out_class = np.ones([NUM_BOXES, 1])*-1
        #         out_score = np.zeros([NUM_BOXES, 1])
        #
        #         out_bboxes[:min(gt_bbox.shape[0], NUM_BOXES)] = gt_bbox[:min(gt_bbox.shape[0], NUM_BOXES)]
        #         out_class[:min(gt_class.shape[0], NUM_BOXES)] = gt_class[:min(gt_class.shape[0], NUM_BOXES)]
        #         out_score[:min(gt_score.shape[0], NUM_BOXES)] = gt_score[:min(gt_score.shape[0], NUM_BOXES)]
        #
        #         results['gt_bboxes']= out_bboxes
        #         results['gt_class']= out_class
        #         results['gt_score']= out_score

        return results

class RandomFlip(object):

    def __init__(self,random_thr=0.5):
        self.random_thr = random_thr

    def __call__(self, results):
        r = random.random()
        # results['ori_img'] = results['img']
        if r >self.random_thr:
            results = self._flip_img(results)
            results = self._flip_boxes(results)

        return results

    def _flip_boxes(self,results):
        bboxes = results['gt_bboxes']
        bboxes[:, [2,0]] = 1 - bboxes[:, [0,2]]
        results['gt_bboxes'] = bboxes

        return results

    def _flip_img(self,results):
        img = results['img']
        img = cv2.flip(img, 1)
        results['img'] = img
        results['img_shape'] = img.shape
        # results['ori_img'] = img
        return results

#随机放射变换
class RandomAffine(object):

    def __init__(self,degrees=10, translate=.1, scale=.5, shear=0.0, border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border

    def __call__(self, results):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]
        img = results['img']
        h,w = results['img_shape'][:2]
        targets = results['gt_bboxes']
        targets[:, [0, 2]] = w * targets[:, [0, 2]]
        targets[:, [1, 3]] = h * targets[:, [1, 3]]
        gt_class = results['gt_class']
        gt_score = results['gt_score']
        if 'mosaic_border' in results:
            self.border = results['mosaic_border']
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2

        # Rotation and Scale 旋转和缩放
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation 平移
        T = np.eye(3)
        T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border[1]  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border[0]  # y translation (pixels)

        # Shear 剪切
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

            targets = targets[i]
            gt_class = gt_class[i]
            gt_score = gt_score[i]
            targets[:, 0:4] = xy[i]
            targets[:, [1, 3]] /= img.shape[0] # height
            targets[:, [0, 2]] /= img.shape[1]  # width
            results['img'] = img
            results['gt_bboxes'] = targets
            results['gt_class'] = gt_class
            results['gt_score'] = gt_score
            results['img_shape'] = img.shape

        return results

#随机平移
class RandomTranslation(object):

    def __init__(self,random_thr=0.5):
        self.random_thr = random_thr

    def __call__(self, results):
        if random.random() < self.rand_thr:
            return results

        img = results['img']
        gt_bboxes = results['gt_bboxes']
        h,w = img.shape[:2]

        gt_bboxes[:, [0,2]] =  w * gt_bboxes[:, [0,2]]
        gt_bboxes[:, [1,3]] = h * gt_bboxes[:, [1,3]]

        x_min,x_max,y_min,y_max = w ,0,h,0
        for bbox in gt_bboxes:
            x_min ,y_min,x_max,y_max= min(x_min, bbox[0]),min(y_min, bbox[1]),max(x_max, bbox[2]),max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in gt_bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        shift_bboxes = np.array(shift_bboxes)
        results['img'] = shift_img
        results['img_shape'] = shift_img.shape
        shift_bboxes[:, [1, 3]] /= shift_img.shape[0]  # height
        shift_bboxes[:, [0, 2]] /= shift_img.shape[1]  # width
        results['gt_bboxes'] =shift_bboxes

        return results

class RandomCrop(object):
    def __init__(self,rand_thr=0.5):
        self.rand_thr = rand_thr

    def __call__(self, results):

        if random.random() < self.rand_thr:
            return results

        img = results['img']
        gt_bboxes = results['gt_bboxes']
        h,w = img.shape[:2]

        gt_bboxes[:, [0,2]] =  w * gt_bboxes[:, [0,2]]
        gt_bboxes[:, [1,3]] = h * gt_bboxes[:, [1,3]]

        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in gt_bboxes:
            x_min ,y_min,x_max,y_max= min(x_min, bbox[0]),min(y_min, bbox[1]),max(x_max, bbox[2]),max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in gt_bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        crop_bboxes = np.array(crop_bboxes)
        results['img'] = crop_img
        results['img_shape'] = crop_img.shape
        crop_bboxes[:, [1, 3]] /= crop_img.shape[0]  # height
        crop_bboxes[:, [0, 2]] /= crop_img.shape[1]  # width
        results['gt_bboxes'] =crop_bboxes
        return results

class MixUp(object):
    def __init__(self,rand_thr=0.5, alpha=1.5, beta=1.5):
        self.alpha = alpha
        self.beta = beta
        self.rand_thr = rand_thr
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))
        self.load_img = LoadImageFromFile()

    def __call__(self, results):
        if np.random.uniform(0., 1.) < self.rand_thr:
            return results

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return results

        truth = results['truth']
        img_ = random.choice(list(truth.keys()))
        true_ = truth[img_]

        result_b = dict(y_true=true_, img_name=img_,img_prefix=results['img_prefix'])
        result_b = self.load_img(result_b)

        img_a = results['img']
        gt_bbox_a = results['gt_bboxes']
        gt_score_a = results['gt_score']
        img_b = result_b['img']
        gt_bbox_b = result_b['gt_bboxes']
        gt_score_b = result_b['gt_score']
        out_img = self._mixup_img(img_a,img_b,factor)
        gt_bbox = np.concatenate([gt_bbox_a, gt_bbox_b], axis=0)

        gt_score = np.concatenate(
            (gt_score_a * factor, gt_score_b * (1. - factor)), axis=0)

        gt_class1 = results['gt_class']
        gt_class2 = result_b['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)


        results['gt_bboxes'] = gt_bbox
        results['gt_class'] = gt_class
        results['gt_score'] = gt_score

        results['img'] = out_img
        results['img_shape'] = out_img.shape

        return results

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')


class Mosaic(object):
    def __init__(self,transforms=dict(),img_scale=608):
        self.img_scale = img_scale
        self.mosaic_border = [-img_scale // 2, -img_scale // 2]

        self.pipeline = Compose(transforms)

    def __call__(self, results):
        s = self.img_scale
        results['mosaic_border'] = self.mosaic_border
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic 中心坐标
        labels4 = []
        gt_classes = []
        gt_scores = []
        truth = results['truth']
        result_0 = results
        imgs = [random.choice(list(truth.keys())) for _ in range(3)]
        result_4 = [result_0] + [dict(y_true=truth[img_], img_name=img_,img_prefix=results['img_prefix']) for img_ in imgs]
        for idx,result in enumerate(result_4):
            result = self.pipeline(result)
            img, (h, w) = result['img'],result['img_shape'][:2]
            if idx == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)大图位置
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)小图位置 从右下角剪切

            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h  # 从左下角剪切
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)  # 从右上角剪切
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)  # 从左上角剪切

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            gt_bboxes = result['gt_bboxes']
            gt_class = result['gt_class']
            gt_score = result['gt_score']
            labels = gt_bboxes.copy()
            if len(labels):  # Normalized xywh to pixel xyxy format
                labels[:, [0,2]] = w * labels[:, [0,2]] + padw  # pad width
                labels[:, [1,3]] = h * labels[:, [1,3]] + padh  # pad height
            labels4.append(labels)
            gt_classes.append(gt_class)
            gt_scores.append(gt_score)

        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            gt_classes = np.concatenate(gt_classes, 0)
            gt_scores = np.concatenate(gt_scores, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4, 0, 2 * s, out=labels4)  # use with random_affine

            _w = labels4[:, 2] - labels4[:, 0]
            _h = labels4[:, 3] - labels4[:, 1]
            area = _w * _h
            labels4 = labels4[np.where(area>20)]
            gt_classes = gt_classes[np.where(area>20)]
            gt_scores = gt_scores[np.where(area>20)]
            labels4[:, [1, 3]] /= 2 * s  # height
            labels4[:, [0, 2]] /= 2 * s  # width

            results['gt_bboxes'] = labels4
            results['gt_class'] = gt_classes
            results['gt_score'] = gt_scores

        results['img'] = img4
        results['img_shape'] = img4.shape

        return results


class Collect(object):
    def __init__(self,keys=['img', 'gt_bboxes', 'gt_class', 'gt_score'],meta_keys=('filename', 'ori_shape', 'img_shape')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            if key == 'img':
                data[key] = results[key]
            else:
                data[key] = results[key].astype(np.float32)
        return data


class Resize(object):
    def __init__(self,img_scale=608,letterbox=True, auto= True,scaleup=True):
        self.img_scale = img_scale
        self.letterbox = letterbox
        self.auto = auto
        self.scaleup = scaleup

    def __call__(self, results):
        if self.letterbox:
            h,w = results['img_shape'][:2]
            img, ratio, pad = letterbox(results['img'],new_shape=self.img_scale,auto=self.auto,scaleup=self.scaleup)
            results['img'] = img
            results['img_shape'] = img.shape
            if 'gt_bboxes' in results:
                gt_bboxes = results['gt_bboxes']
                if len(gt_bboxes)>0:
                    gt_bboxes[:, [0,2]] = ratio[0] * w * gt_bboxes[:, [0,2]] + pad[0]  # pad width
                    gt_bboxes[:, [1,3]] = ratio[1] * h * gt_bboxes[:, [1,3]] + pad[1]  # pad height
                    # gt_bboxes[:, 2] = ratio[0] * w * gt_bboxes[:, 2] + pad[0]
                    # gt_bboxes[:, 3] = ratio[1] * h * gt_bboxes[:, 3] + pad[1]
                    gt_bboxes[:, [1, 3]] /= img.shape[0]  # height
                    gt_bboxes[:, [0, 2]] /= img.shape[1]  # width
                results['gt_bboxes'] = gt_bboxes
        else:
            results = self._resize_img(results)
        return results

    def _resize_img(self, results):
        try:
            img = results['img']
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_scale / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            results['img'] = img
            results['img_shape'] = img.shape
        except Exception as e:
            print(results)
            raise e
        return results

def draw_box(img, bboxes,class_name,gt_class):
    # for b in bboxes:
    #     img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    for idx,ind in enumerate(gt_class):
        if ind>-1:
            b = bboxes[idx]
            img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img,class_name[int(ind)],(b[0], b[1]), font, 0.3, (0, 0, 255), 1)
    return img


class RandomShape(object):
    #Only multiples of 32 are supported
    def __init__(self, sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608], random_inter=True):

        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []

    def __call__(self,batch):
        shape = np.random.choice(self.sizes)

        method = np.random.choice(self.interps) if self.random_inter else cv2.INTER_NEAREST
        for i,result in enumerate(batch):
            im = result['img']
            if isinstance(im,torch.Tensor):
                im = im.numpy()
                if im.shape[0]==3:
                    im = im.transpose(1, 2, 0)
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h

            bboxes = result['gt_bboxes']
            bboxes[:, 0] *= scale_x
            bboxes[:, 2] *= scale_x
            bboxes[:, 1] *= scale_y
            bboxes[:, 3] *= scale_y
            x,y = np.where(bboxes < 0)
            if len(x)>0:
                for i in x:
                    print('bbox index < 0 :',bboxes[i])

            result['gt_bboxes'] = bboxes
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)

            # tmp = im*255
            # a = draw_box(tmp.copy(), bboxes.astype(np.int32), ['penghu','1','2','3','4','5'], result['gt_class'])
            # import os
            # import uuid
            # cv2.imwrite(os.path.join('/disk2/project/test/v2.0/yolov5/dataset/123/test', str(uuid.uuid1()) + '.jpg'), a)
            if len(im.shape) < 3:
                im = np.expand_dims(im, -1)
            result['img_metas']['img_shape'] = im.shape
            im = to_tensor(im.transpose(2, 0, 1))
            result['img'] = im
        return batch



def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = np.float32(img) if img.dtype != np.float32 else img.copy()
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)