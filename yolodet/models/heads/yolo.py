#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/10/13 15:45
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: yolo.py
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
from abc import ABCMeta, abstractmethod

import torch.nn as nn

import numpy as np
import torch

from yolodet.models.loss.base import bbox_iou, box_iou
from yolodet.models.utils.torch_utils import initialize_weights
from yolodet.utils.util import multi_apply
from yolodet.models.heads.base import xyxy2xywh, nms_cpu, soft_nms_cpu, Y, scale_coords,non_max_suppression
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.utils.registry import LOSS


class BaseHead(nn.Module,metaclass=ABCMeta):

    def __init__(self, label_smooth=True,conf_balances=[0.4,1,4], deta=0.01, anchors=[[12, 16], [19, 36], [40, 28],
                                                              [36, 75], [76, 55], [72, 146],
                                                              [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], downsample_ratios=[32, 16, 8],bbox_weight = False,yolo_loss_type=None,
                 in_channels=[128, 256, 512], num_classes=80,nms_type='nms', nms_thr=.5, ignore_thre=.3,anchor_t=4,
                 bbox_loss=dict(type='IOU_Loss', iou_type='CIOU'), confidence_loss=dict(type='Conf_Loss'),
                 class_loss=dict(type='Class_Loss'),norm_type='BN',num_groups=None):
        super(BaseHead, self).__init__()

        assert yolo_loss_type in (None,'yolov4','yolov5'),'use_yolo_loss just support [None,yolov4,yolov5]'
        self.yolo_loss_type = yolo_loss_type
        self.conf_balances = conf_balances
        self.num_classes = num_classes
        assert nms_type in ['nms','soft_nms'],'nms type only support [nms,soft_nms],other has not implementation'
        self.nms_thr = nms_thr
        self.nms_type = nms_type
        self.out_channels = []
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.base_num = 5
        self.bbox_weight = bbox_weight
        self.anchor_t=anchor_t
        for mask in anchor_masks:
            self.out_channels.append(len(mask) * (self.base_num + num_classes))

        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        if label_smooth is None or not isinstance(label_smooth, bool):
            label_smooth = False
        self.label_smooth = label_smooth
        self.deta = deta
        self.ignore_thre = ignore_thre
        self.bbox_loss = build_from_dict(bbox_loss, LOSS)
        self.confidence_loss = build_from_dict(confidence_loss, LOSS)
        self.class_loss = build_from_dict(class_loss, LOSS)

    @abstractmethod
    def forward(self, x):
        pass

    def get_target(self, pred, img_metas, batch_size, gt_bbox, gt_class, gt_score):
        if self.yolo_loss_type == 'yolov5':
            return self.get_yolov5_target(pred, img_metas, batch_size, gt_bbox, gt_class, gt_score)
        else:
            return self.get_yolov4_target(pred, img_metas, batch_size, gt_bbox, gt_class, gt_score)

    def loss_single(self, pred, indices,tbox,tcls,anchors, conf_balances,ignore_mask):
        if self.yolo_loss_type == 'yolov5':
            return self.yolov5_loss_single(pred, indices,tbox,tcls,anchors, conf_balances,ignore_mask)
        else:
            return self.yolov4_loss_single(pred, indices,tbox,tcls,anchors, conf_balances,ignore_mask)

    def loss(self, input, img_metas, batch_size, gt_bboxes, gt_class, gt_score):
        pred = self.get_pred(input)
        indices,tbox,tcls,ancher,ignore_mask = self.get_target(pred = pred,img_metas=img_metas, batch_size=batch_size, gt_bbox=gt_bboxes, gt_class=gt_class, gt_score=gt_score)

        bbox_loss, confidence_loss, class_loss = multi_apply(self.loss_single, pred, indices,tbox,tcls,ancher,self.conf_balances,ignore_mask)

        return dict(bbox_loss=bbox_loss, confidence_loss=confidence_loss, class_loss=class_loss)

    def yolov4_loss_single(self, pred, indices,tbox,tcls,anchors, conf_balances,ignore_mask):
        device = pred.device
        b, a, gj, gi = indices
        # 2.0 - (1.0 * target[..., 2:3] * target[..., 3:4]) / (fsize ** 2)
        tobj = torch.zeros_like(pred[..., 4]).to(device)  # target obj
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        gird_h,gird_w = pred.shape[-3:-1]
        nb = b.shape[0]  # number of targets
        if nb:
            ps = pred[b,a,gj, gi]  # prediction subset corresponding to targets
            tobj[b,a,gj, gi] = 1
            # bbox loss
            pxy = ps[:, :2].sigmoid() # -0.5<pxy<1.5
            pwh = torch.exp(ps[..., 2:4]) * torch.from_numpy(anchors).to(device)
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            weight = torch.ones_like(ps[..., 4]).to(device)  # target obj
            if self.bbox_weight:
                weight = (2.0 - (1.0 * tbox[..., 2:3] * tbox[..., 3:4]) / (gird_h * gird_w)).reshape(-1)
            lbox += self.bbox_loss(pbox,tbox,weight)[0]

            t = torch.full_like(ps[:, self.base_num:], 0).to(device)  # targets
            t[range(nb), tcls.squeeze()] = 1
            if self.label_smooth:#https://arxiv.org/pdf/1812.01187.pdf
                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                smooth_onehot = t * (1 - self.deta) + self.deta *torch.from_numpy(uniform_distribution).to(device).float()
                t = smooth_onehot
            lcls += self.class_loss(ps[:, self.base_num:], t,self.num_classes)

        lobj += self.confidence_loss(pred[..., 4], tobj,ignore_mask)*conf_balances

        return lbox, lobj, lcls

    def get_pred(self, preds):
        preds_result = []
        for index,mask in enumerate(self.anchor_masks):
            pred = preds[index]
            batch = pred.shape[0]
            grid_h, grid_w = pred.shape[-2:]
            out_ch = self.base_num + self.num_classes
            pred = pred.view(batch, len(mask), out_ch, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2)  # batch,size,size，mask,out_ch
            preds_result.append(pred)
        return preds_result

    def get_yolov4_eval_bboxes(self,x):
        z = []  # inference output
        for i, mask in enumerate(self.anchor_masks):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, len(mask), self.base_num + self.num_classes, ny, nx).permute(0, 1, 3, 4,
                                                                                              2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i]
            y[..., :2] = (torch.sigmoid(y[..., :2]) + grid_xy) * self.downsample_ratios[i]  # 映射回原图位置
            y[..., 2:4] = torch.exp(y[..., 2:4]) * torch.from_numpy(np.array([self.anchors[idx] for idx in mask])).to(
                y.device).view(1, -1, 1, 1, 2)  # 映射回原图wh尺寸
            y[..., 4:] = torch.sigmoid(y[..., 4:])
            z.append(y.view(bs, -1, self.base_num + self.num_classes))

        return torch.cat(z, 1), x

    def get_yolov5_eval_bboxes(self,x):
        z = []  # inference output
        for i, mask in enumerate(self.anchor_masks):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, len(mask), self.base_num + self.num_classes, ny, nx).permute(0, 1, 3, 4,
                                                                                              2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_xy) * self.downsample_ratios[i]  # xy 映射回原图位置
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.from_numpy(np.array([self.anchors[idx] for idx in mask])).to(
                y.device).view(1, -1, 1, 1, 2)  # 映射回原图wh尺寸
            z.append(y.view(bs, -1, self.base_num + self.num_classes))

        return torch.cat(z, 1), x


    def yolov5_loss_single(self, pred, indices,tbox,tcls,anchors, conf_balances,ignore_mask=None):

        device = pred.device
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
            # return positive, negative label smoothing BCE targets
            return 1.0 - 0.5 * eps, 0.5 * eps

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        eps = 0
        if self.label_smooth and self.deta>0:
            eps = self.deta
        cp, cn = smooth_BCE(eps=eps)

        # per output
        nt = 0  # number of targets
        pi = pred
        b, a, gj, gi = indices # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # -0.5<pxy<1.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * torch.from_numpy(anchors).to(device)  # 0-4倍缩放 model.hyp['anchor_t']=4
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            bbox_loss,giou= self.bbox_loss(pbox,tbox)
            lbox += bbox_loss

            # Obj
            # tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
            tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            t = torch.full_like(ps[:, self.base_num:], cn).to(device)  # targets
            # t[range(nb), tcls] = cp
            t[range(nb), tcls.squeeze()] = cp
            lcls += self.class_loss(ps[:, self.base_num:], t,self.num_classes)

        lobj += self.confidence_loss(pi[..., 4], tobj) * conf_balances  # obj loss

        return lbox, lobj, lcls

    def get_eval_bboxes(self,x):
        if self.yolo_loss_type == 'yolov5':
            return self.get_yolov5_eval_bboxes(x)
        else:
            return self.get_yolov4_eval_bboxes(x)

    def get_det_bboxes(self, x, img_metas,kwargs):
        # pred = self.get_eval_bboxes(x)[0]
        return self.get_nms_result(img_metas, kwargs, x)

    def get_nms_result(self, img_metas, kwargs, pred):

        scores_thr,merge = 0.3,False
        if 'scores_thr' in kwargs.keys():
            scores_thr = kwargs['scores_thr']
        if 'merge' in kwargs.keys():
            merge = kwargs['merge']

        det_result = []
        pred = non_max_suppression(pred, conf_thres=scores_thr, iou_thres=self.nms_thr,merge=merge)
        ori_shape = img_metas['ori_shape']
        img_shape = img_metas['img_shape']
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_shape, det[:, :4], ori_shape).round()
                # Write results
                for *xyxy, conf, cls in det:
                    label = int(cls)
                    score = float(conf)
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    object_dic = dict()

                    object_dic['label'] = label
                    object_dic['score'] = score
                    object_dic['bbox'] = bbox
                    det_result.append(object_dic)
        return det_result

    # 为了简便起见，直接转换数据格式为yolov5 build_target需要的格式，值得注意的是yolov5为了增加训练正样本数量（yolo原始方法一个目标最多由1个grid预测（中心点cx,cy所在位置grid），youlov5采用中心点（cx,cy）所在grid+附件的两个grid一起预测。）
    # 针对预测grid位置判定会根据cx,cy偏移中心点（0.5）位置，多添加两个grid预测位置。在x轴方向cx偏移量>0.5,添加grid+1的位置预测，cx偏移量<0.5,添加grid-1的位置预测。y轴同理
    # 由于添加两个grid预测位置，所以 gt_xy的预测偏移范围为-0.5<pxy<1.5,损失函数定义需要在此范围内。
    # 针对anchor的选择不再是通过IOU找到最合适的anchor方式。二是采用gt框/anchor小于4,(也就是，gt和anchor的缩放比例在4以内都可预测)，预测的缩放尺度为0<pwh<4 ，损失函数定义时候请注意
    def get_yolov5_target(self, pred, img_metas, batch_size, gt_bbox, gt_class, gt_score):
        device = pred[0].device
        gain = torch.ones(6, device=device)  # normalized to gridspace gain
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        targets = ft([]).to(device)
        for i, gtb in enumerate(gt_bbox):
            gtc = torch.from_numpy(gt_class[i]).to(device)
            img_idx = torch.ones(len(gtb), 1, device=device) * i
            targets = torch.cat((targets, torch.cat((img_idx, gtc, torch.from_numpy(gtb).to(device)), dim=-1)))
        na, nt = len(self.anchor_masks), len(targets)
        tcls, tbox, indices, anch,ignore_mask = [], [], [], [],[]
        targets[..., 2:] = xyxy2xywh(targets[..., 2:])
        g = 0.5  # offset grid中心偏移
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=device).float()  # overlap offsets 按grid区域换算偏移区域， 附近的4个网格 上下左右
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
        for idx, (mask, downsample_ratio) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            anchors = np.array(self.anchors, dtype=np.float32)[mask] / downsample_ratio  # Scale
            # for i in range(len(self.anchor_masks)):
            #     anchors = self.anchors[i]
            gain[2:] = torch.tensor(pred[idx].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / torch.from_numpy(anchors[:, None]).to(device)  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter t为过滤后所有匹配锚框缩放尺度小于4的真框 a 位置信息

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                # j,k 为小于0.5的偏移 ，l,m为大于0.5的偏移
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]),
                                                                            0)  # t 原始target, t[j] x<.5 偏移的target, t[k] y<.5 偏移的target, t[l] x>.5 偏移的target, t[m] y>.5 偏移的target
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]),
                                    0) * g  # z 原始target,x<0.5 +0.5 ,y<0.5 +0.5,x>.5 -0.5,y>.5 -0.5

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 获取所有的grid 位置 -0.5<offsets<0.5
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box x,y 偏移范围在[-0.5,1.5]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            ignore_mask.append([])

        return indices, tbox, tcls, anch, ignore_mask


    def get_yolov4_target(self, pred, img_metas, batch_size, gt_bbox, gt_class, gt_score):
        device = pred[0].device
        h, w = img_metas[0]['img_shape'][:2]
        tcls, tbox, indices ,ignore_mask,anch= [], [], [],[],[]
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        lt = torch.cuda.LongTensor if pred[0].is_cuda else torch.Tensor

        for index, (mask, downsample_ratio) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            b,a,gj,gi,gxywh = lt([]).to(device),lt([]).to(device),lt([]).to(device),lt([]).to(device),ft([]).to(device)
            cls = lt([]).to(device)
            anchors = np.array(self.anchors, dtype=np.float32)[mask]/downsample_ratio  # Scale
            batch_ignore_mask = torch.ones((batch_size,len(mask), int(h / downsample_ratio),
                                              int(w / downsample_ratio), 1)).to(device)  # large object
            for bs in range(batch_size):
                xywh = xyxy2xywh(gt_bbox[bs]) if isinstance(gt_bbox[bs], torch.Tensor) else xyxy2xywh(torch.from_numpy(gt_bbox[bs]).to(device))
                if len(xywh) == 0:
                    continue

                grid_h,grid_w = int(h / downsample_ratio),int(w / downsample_ratio)

                all_anchors_grid = np.array(self.anchors, dtype=np.float32) / downsample_ratio  # Scale

                ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
                ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
                ref_anchors = torch.from_numpy(ref_anchors)  # [0,0,anchor_w,anchor_h]

                gt = xywh*torch.tensor(([grid_w,grid_h,grid_w,grid_h])).to(device).float() # x,y ,w, h，Scale
                score ,_cls = gt_score[bs],gt_class[bs]

                cx_grid = gt[:, 0].floor().cpu().numpy()  # grid_x grid_y
                cy_grid = gt[:, 1].floor().cpu().numpy()  # grid_y
                n = len(gt)
                truth_box = torch.zeros(n, 4)
                truth_box[:n, 2:4] = gt[:n, 2:4]
                anchor_ious = box_iou(truth_box, ref_anchors)

                best_n_all = anchor_ious.argmax(dim=1)  # 返回按行比较最大值的位置
                best_n = best_n_all % 3
                best_n_mask = ((best_n_all == mask[0]) | (best_n_all == mask[1]) | ( best_n_all == mask[2]))  # 查看是否和当前尺度有最大值得IOU交集，如果有为1，否则为0

                if sum(best_n_mask) == 0:  # 如果和当前尺度不是最大IOU交集，返回
                    continue

                truth_box[:n, 0:2] = gt[:n, 0:2]  # cx 包含位置和偏移量，整数位代表坐标位置，小数位代表偏移量
                # truth_box[:n, 1] = gt[:n, 1]  # cy 包含位置和偏移量，整数位代表坐标位置，小数位代表偏移量

                single_ignore_mask = np.zeros(( len(mask),grid_h, grid_w, 1), dtype=np.float32)

                pred_ious = box_iou(pred[index][bs,...,:4].reshape(-1,4), truth_box.reshape(-1,4).to(device), xyxy=False)  # truth框和基本锚框的IOU，含位置信息
                pred_best_iou, _ = pred_ious.max(dim=1)  # [最大值，索引]
                pred_best_iou = (pred_best_iou > self.ignore_thre)  # 过滤掉小于阈值的数据，大于阈值1，小于0
                pred_best_iou = pred_best_iou.view(single_ignore_mask.shape)  # 映射到具体位置，是否有目标,1代表有目标物，0代表没有目标物
                # set mask to zero (ignore) if pred matches truth
                single_ignore_mask = ~ pred_best_iou  # 取反，为未包含目标的框位置，1代表没有目标物，0代表有目标物

                # torch.ones(len(truth_box))[best_n_mask].to(device)
                b =  torch.cat((b,torch.ones(len(truth_box))[best_n_mask].long().to(device)*bs))
                a =  torch.cat((a, best_n[best_n_mask].to(device).long()))
                gi =  torch.cat((gi, torch.from_numpy(cx_grid)[best_n_mask].to(device).long()))
                gj =  torch.cat((gj, torch.from_numpy(cy_grid)[best_n_mask].to(device).long()))
                gxywh =  torch.cat((gxywh,truth_box[best_n_mask].to(device)))
                cls = torch.cat((cls, torch.from_numpy(_cls)[best_n_mask].to(device).long()))
                single_ignore_mask[a,gj,gi] = 0
                # ignore_mask[gj, gi, a] = 0
                batch_ignore_mask[bs,:] = single_ignore_mask

            indices.append((b, a, gj, gi))
            gxywh[...,:2] = gxywh[...,:2] - gxywh[...,:2].long()
            tbox.append(gxywh)
            tcls.append(cls)
            anch.append(anchors[a.cpu().numpy()])  # anchors
            ignore_mask.append(batch_ignore_mask)

        return indices, tbox, tcls, anch, ignore_mask

    def init_weights(self):
        initialize_weights(self)





    # def get_target_2(self,pred,img_metas, batch_size, gt_bbox, gt_class, gt_score):
    #     device = pred[0].device
    #     h, w = img_metas[0]['img_shape'][:2]
    #     # downsampling ratio 32 16 8 ,grid form
    #     batch_s_ignore_mask = torch.ones((batch_size, int(h / self.downsample_ratios[2]),
    #                                     int(w / self.downsample_ratios[2]), len(self.anchor_masks[0]),1)).to(device)  # small object
    #     batch_m_ignore_mask = torch.ones((batch_size, int(h / self.downsample_ratios[1]),
    #                                     int(w / self.downsample_ratios[1]), len(self.anchor_masks[0]),1)).to(device)   # middle object
    #     batch_l_ignore_mask = torch.ones((batch_size, int(h / self.downsample_ratios[0]),
    #                                     int(w / self.downsample_ratios[0]), len(self.anchor_masks[0]),1)).to(device)   # large object
    #
    #     batch_true_sbbox = torch.zeros((batch_size, int(h / self.downsample_ratios[2]),
    #                                     int(w / self.downsample_ratios[2]), len(self.anchor_masks[0]),
    #                                     5 + self.num_classes)).to(device)  # small object
    #     batch_true_mbbox = torch.zeros((batch_size, int(h / self.downsample_ratios[1]),
    #                                     int(w / self.downsample_ratios[1]), len(self.anchor_masks[0]),
    #                                     5 + self.num_classes)).to(device)   # middle object
    #     batch_true_lbbox = torch.zeros((batch_size, int(h / self.downsample_ratios[0]),
    #                                     int(w / self.downsample_ratios[0]), len(self.anchor_masks[0]),
    #                                     5 + self.num_classes)).to(device)
    #
    #     batch_true_label = [batch_true_lbbox, batch_true_mbbox, batch_true_sbbox]
    #     batch_ignore_mask = [batch_l_ignore_mask, batch_m_ignore_mask, batch_s_ignore_mask]
    #
    #     gt_xywh = xyxy2xywh(gt_bbox)  # x,y ,w, h
    #
    #     for bs in range(batch_size):
    #
    #         for index, (mask, downsample_ratio) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
    #             grid_h = int(h / downsample_ratio)
    #             grid_w = int(w / downsample_ratio)
    #             target = np.zeros((grid_h, grid_w, len(mask), 5 + self.num_classes), dtype=np.float32)
    #             ignore_mask = np.zeros((grid_h, grid_w, len(mask), 1), dtype=np.float32)
    #
    #             all_anchors_grid = np.array(self.anchors, dtype=np.float32) / downsample_ratio  # Scale
    #
    #             ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
    #             ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
    #             ref_anchors = torch.from_numpy(ref_anchors)  # [0,0,anchor_w,anchor_h]
    #
    #             gt = gt_xywh[bs]*torch.tensor(([grid_w,grid_h,grid_w,grid_h])).to(device).double() # x,y ,w, h，Scale
    #             score = gt_score[bs]
    #             cls = gt_class[bs].cpu().numpy()
    #             cx_grid = gt[:, 0].floor().cpu().numpy()  # grid_x
    #             cy_grid = gt[:, 1].floor().cpu().numpy()  # grid_y
    #             n_gt = (gt.sum(dim=-1) > 0).sum(dim=-1)
    #             n = int(n_gt)
    #             if n == 0:
    #                 continue
    #             truth_box = torch.zeros(n, 4)
    #             truth_box[:n, 2:4] = gt[:n, 2:4]
    #             anchor_ious = bbox_iou(truth_box, ref_anchors)
    #
    #             best_n_all = anchor_ious.argmax(dim=1)  # 返回按行比较最大值的位置
    #             best_n = best_n_all % 3
    #             best_n_mask = ((best_n_all == mask[0]) | (best_n_all == mask[1]) | (
    #                     best_n_all == mask[2]))  # 查看是否和当前尺度有最大值得IOU交集，如果有为1，否则为0
    #
    #             if sum(best_n_mask) == 0:  # 如果和当前尺度不是最大IOU交集，返回
    #                 continue
    #
    #             truth_box[:n, 0] = gt[:n, 0]  # cx 包含位置和偏移量，整数位代表坐标位置，小数位代表偏移量
    #             truth_box[:n, 1] = gt[:n, 1]  # cy 包含位置和偏移量，整数位代表坐标位置，小数位代表偏移量
    #
    #             pred_ious = bbox_iou(pred[index][bs,...,:4].reshape(-1,4), truth_box.to(pred[index].device), x1y1x2y2=False)  # truth框和基本锚框的IOU，含位置信息
    #             pred_best_iou, _ = pred_ious.max(dim=1)  # [最大值，索引]
    #             pred_best_iou = (pred_best_iou > self.ignore_thre)  # 过滤掉小于阈值的数据，大于阈值1，小于0
    #             pred_best_iou = pred_best_iou.view(ignore_mask.shape)  # 映射到具体位置，是否有目标,1代表有目标物，0代表没有目标物
    #             # set mask to zero (ignore) if pred matches truth
    #             ignore_mask = ~ pred_best_iou  # 取反，为未包含目标的框位置，1代表没有目标物，0代表有目标物
    #
    #
    #             for ti in range(best_n.shape[0]):  # 迭代每个true bbox
    #                 if best_n_mask[ti] == 1:  # 如果该true bbox 在目前尺度下，执行如下代码
    #                     try:
    #                         i, j = int(cx_grid[ti]), int(cy_grid[ti])  # cx ，cy,框的中心位置
    #                         a = best_n[ti]  # 基础锚框的索引值
    #                         target[j, i, a, 0] = truth_box[ti, 0]
    #                         target[j, i, a, 1] = truth_box[ti, 1]
    #                         target[j, i, a, 2] = truth_box[ti, 2]
    #                         target[j, i, a, 3] = truth_box[ti, 3]
    #                         # objectness record gt_score
    #                         target[j, i, a, 4] = score[ti]
    #                         ignore_mask[j,i,a] = 0
    #
    #                         # classification
    #                         onehot = np.zeros(self.num_classes, dtype=np.float)
    #                         onehot[int(cls[ti][0])] = 1.0
    #                         if self.label_smooth:
    #                             uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
    #                             deta = self.deta
    #                             smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
    #                             target[j, i, a, 5:] = smooth_onehot
    #                         else:
    #                             target[j, i, a, 5:] = onehot
    #                     except:
    #                         print('target size :',target.size())
    #                         print('i:{},j:{}'.format(i,j))
    #                         pass
    #             batch_true_label[index][bs, :, :, :, :] = torch.from_numpy(target)
    #             batch_ignore_mask[index][bs,:] = ignore_mask
    #     return batch_true_label,batch_ignore_mask
