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
import torch
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod
from yolodet.utils.registry import LOSS
from yolodet.utils.util import multi_apply
from yolodet.models.loss.base import box_iou
from yolodet.utils.newInstance_utils import build_from_dict
from yolodet.models.utils.torch_utils import initialize_weights
from yolodet.models.heads.base import xyxy2xywh, scale_coords, non_max_suppression


class BaseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 label_smooth=True,
                 conf_balances=[0.4, 1, 4],
                 deta=0.01,
                 anchors=[[12, 16], [19, 36], [40, 28],
                          [36, 75], [76, 55], [72, 146],
                          [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 downsample_ratios=[32, 16, 8],
                 bbox_weight=False,
                 yolo_loss_type=None,
                 in_channels=[128, 256, 512],
                 num_classes=80,
                 nms_type='nms',
                 nms_thr=.5,
                 ignore_thre=.3,
                 anchor_t=4,
                 bbox_loss=dict(type='IOU_Loss', iou_type='CIOU'),
                 confidence_loss=dict(type='Conf_Loss'),
                 class_loss=dict(type='cls_loss'),
                 norm_type='BN',
                 num_groups=None):

        super(BaseHead, self).__init__()
        assert yolo_loss_type in (None, 'yolov4', 'yolov5'), \
            'use_yolo_loss just support [None,yolov4,yolov5]'
        self.yolo_loss_type = yolo_loss_type
        self.conf_balances = conf_balances
        self.num_classes = num_classes
        assert nms_type in ['nms', 'soft_nms'], \
            'nms type only support [nms,soft_nms],other has not implementation'
        self.nms_thr = nms_thr
        self.nms_type = nms_type
        self.out_channels = []
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.base_num = 5
        self.bbox_weight = bbox_weight
        self.anchor_t = anchor_t
        for mask in anchor_masks:
            self.out_channels.append(len(mask) * (self.base_num + num_classes))

        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.down_ratios = downsample_ratios
        if label_smooth is None or not isinstance(label_smooth, bool):
            label_smooth = False
        self.label_smooth = label_smooth
        self.deta = deta
        self.ignore_thre = ignore_thre
        self.bbox_loss = build_from_dict(bbox_loss, LOSS)
        self.conf_loss = build_from_dict(confidence_loss, LOSS)
        self.cls_loss = build_from_dict(class_loss, LOSS)

    @abstractmethod
    def forward(self, x):
        pass

    def get_target(self,
                   pred,
                   img_metas,
                   batch_size,
                   gt_bbox,
                   gt_class,
                   gt_score):

        if self.yolo_loss_type == 'yolov5':
            return self.get_yolov5_target(pred, img_metas, batch_size,
                                          gt_bbox, gt_class, gt_score)
        else:
            return self.get_yolov4_target(pred, img_metas, batch_size,
                                          gt_bbox, gt_class, gt_score)

    def loss_single(self,
                    pred,
                    indices,
                    tbox,
                    tcls,
                    anchors,
                    conf_balances,
                    ignore_mask):

        if self.yolo_loss_type == 'yolov5':
            return self.yolov5_loss_single(pred, indices, tbox, tcls,
                                           anchors, conf_balances, ignore_mask)
        else:
            return self.yolov4_loss_single(pred, indices, tbox, tcls,
                                           anchors, conf_balances, ignore_mask)

    def loss(self,
             input,
             img_metas,
             batch_size,
             gt_bboxes,
             gt_class,
             gt_score):

        pred = self.get_pred(input)
        indices, tbox, tcls, \
        ancher, ignore_mask = self.get_target(pred=pred,
                                              img_metas=img_metas,
                                              batch_size=batch_size,
                                              gt_bbox=gt_bboxes,
                                              gt_class=gt_class,
                                              gt_score=gt_score)

        bbox_loss, conf_loss, cls_loss = multi_apply(self.loss_single, pred,
                                                     indices, tbox, tcls, ancher,
                                                     self.conf_balances, ignore_mask)

        return dict(bbox_loss=bbox_loss, conf_loss=conf_loss, cls_loss=cls_loss)

    def yolov4_loss_single(self,
                           pred,
                           indices,
                           tbox,
                           tcls,
                           anchors,
                           conf_balances,
                           ignore_mask):

        device = pred.device
        b, a, gj, gi = indices
        # 2.0 - (1.0 * target[..., 2:3] * target[..., 3:4]) / (fsize ** 2)
        tobj = torch.zeros_like(pred[..., 4]).to(device)  # target obj
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        gird_h, gird_w = pred.shape[-3:-1]
        nb = b.shape[0]  # number of targets
        if nb:
            ps = pred[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1
            # bbox loss
            pxy = ps[:, :2].sigmoid()  # -0.5<pxy<1.5
            pwh = torch.exp(ps[..., 2:4]) * torch.from_numpy(anchors).to(device)
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            weight = torch.ones_like(ps[..., 4]).to(device)  # target obj
            if self.bbox_weight:
                weight = (2.0 - (1.0 * tbox[..., 2:3] *
                          tbox[..., 3:4]) / (gird_h * gird_w)).reshape(-1)
            lbox += self.bbox_loss(pbox, tbox, weight)[0]

            t = torch.full_like(ps[:, self.base_num:], 0).to(device)  # targets
            t[range(nb), tcls.squeeze()] = 1
            if self.label_smooth:
                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                smooth_onehot = t * (1 - self.deta) + self.deta * \
                                torch.from_numpy(uniform_distribution).to(device).float()
                t = smooth_onehot
            lcls += self.cls_loss(ps[:, self.base_num:], t, self.num_classes)

        lobj += self.conf_loss(pred[..., 4], tobj, ignore_mask) * conf_balances

        return lbox, lobj, lcls

    def get_pred(self, preds):
        preds_result = []
        for index, mask in enumerate(self.anchor_masks):
            pred = preds[index]
            batch = pred.shape[0]
            grid_h, grid_w = pred.shape[-2:]
            out_ch = self.base_num + self.num_classes
            pred = pred.view(batch, len(mask), out_ch, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2)  # batch,size,size,mask,out_ch
            preds_result.append(pred)
        return preds_result

    def get_yolov4_eval_bboxes(self, x):
        z = []  # inference output
        for i, mask in enumerate(self.anchor_masks):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, len(mask), self.base_num +
                             self.num_classes, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i]
            y[..., :2] = (torch.sigmoid(y[..., :2]) + grid_xy) * self.down_ratios[i]  # 映射回原图位置
            y[..., 2:4] = torch.exp(y[..., 2:4]) * \
                          torch.from_numpy(np.array([self.anchors[idx]
                          for idx in mask])).to(y.device).view(1, -1, 1, 1, 2)  # 映射回原图wh尺寸
            y[..., 4:] = torch.sigmoid(y[..., 4:])
            z.append(y.view(bs, -1, self.base_num + self.num_classes))

        return torch.cat(z, 1), x

    def get_yolov5_eval_bboxes(self, x):
        z = []  # inference output
        for i, mask in enumerate(self.anchor_masks):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, len(mask), self.base_num + self.num_classes,
                             ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid_xy = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i].sigmoid()
            # xy 映射回原图位置
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_xy) * self.down_ratios[i]
            # 映射回原图wh尺寸
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.from_numpy(
                           np.array([self.anchors[idx] for idx in mask])).to(
                           y.device).view(1, -1, 1, 1, 2)
            z.append(y.view(bs, -1, self.base_num + self.num_classes))

        return torch.cat(z, 1), x

    def yolov5_loss_single(self,
                           pred,
                           indices,
                           tbox,
                           tcls,
                           anchors,
                           conf_balances,
                           ignore_mask=None):

        device = pred.device
        ft = torch.cuda.FloatTensor if pred.is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)

        # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        def smooth_BCE(eps=0.1):
            # return positive, negative label smoothing BCE targets
            return 1.0 - 0.5 * eps, 0.5 * eps

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        eps = 0
        if self.label_smooth and self.deta > 0:
            eps = self.deta
        cp, cn = smooth_BCE(eps=eps)

        # per output
        nt = 0  # number of targets
        pi = pred
        b, a, gj, gi = indices  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # iou loss: GIoU
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # -0.5<pxy<1.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * torch.from_numpy(anchors).to(
                device)  # 0-4倍缩放 model.hyp['anchor_t']=4
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            bbox_loss, giou = self.bbox_loss(pbox, tbox)
            lbox += bbox_loss

            # conf loss: contain giou ratio
            # tobj[b, a, gj, gi] = (1.0 - model.gr) +
            # model.gr * giou.detach().clamp(0).type(tobj.dtype)
            tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # cls loss
            t = torch.full_like(ps[:, self.base_num:], cn).to(device)  # targets
            # t[range(nb), tcls] = cp
            t[range(nb), tcls.squeeze()] = cp
            lcls += self.cls_loss(ps[:, self.base_num:], t, self.num_classes)

        lobj += self.conf_loss(pi[..., 4], tobj) * conf_balances  # obj loss

        return lbox, lobj, lcls

    def get_eval_bboxes(self, x):
        if self.yolo_loss_type == 'yolov5':
            return self.get_yolov5_eval_bboxes(x)
        else:
            return self.get_yolov4_eval_bboxes(x)

    def get_det_bboxes(self, x, img_metas, kwargs):
        # pred = self.get_eval_bboxes(x)[0]
        return self.get_nms_result(img_metas, kwargs, x)

    def get_nms_result(self, img_metas, kwargs, pred):

        scores_thr, merge = 0.3, False
        if 'scores_thr' in kwargs.keys():
            scores_thr = kwargs['scores_thr']
        if 'merge' in kwargs.keys():
            merge = kwargs['merge']

        det_result = []
        pred = non_max_suppression(pred, conf_thres=scores_thr,
                                   iou_thres=self.nms_thr, merge=merge)
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

    # 为了简便起见,直接转换数据格式为yolov5 build_target需要的格式,值得注意的是yolov5为了增加训练正样本数量
    # (yolo原始方法一个目标最多由1个grid预测(中心点cx,cy所在位置grid),youlov5采用中心点(cx,cy)
    # 所在grid+附件的两个grid一起预测。),针对预测grid位置判定会根据cx,cy偏移中心点(0.5)位置,多添加两个
    # grid预测位置。在x轴方向cx偏移量>0.5,添加grid+1的位置预测,cx偏移量<0.5,添加grid-1的位置预测。y轴同理
    # 由于添加两个grid预测位置,所以 gt_xy的预测偏移范围为-0.5<pxy<1.5,损失函数定义需要在此范围内。
    # 针对anchor的选择不再是通过IOU找到最合适的anchor方式。二是采用gt框/anchor小于4,(也就是,gt和anchor的
    # 缩放比例在4以内都可预测),预测的缩放尺度为0<pwh<4,损失函数定义时候请注意
    def get_yolov5_target(self,
                          pred,
                          img_metas,
                          batch_size,
                          gt_bbox,
                          gt_class,
                          gt_score):
        device = pred[0].device
        gain = torch.ones(6, device=device)  # normalized to gridspace gain
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        targets = ft([]).to(device)
        for i, gtb in enumerate(gt_bbox):
            gtc = torch.from_numpy(gt_class[i]).to(device)
            img_idx = torch.ones(len(gtb), 1, device=device) * i
            targets = torch.cat((targets, torch.cat((img_idx, gtc,
                                torch.from_numpy(gtb).to(device)), dim=-1)))
        na, nt = len(self.anchor_masks), len(targets)
        tcls, tbox, indices, anch, ignore_mask = [], [], [], [], []
        targets[..., 2:] = xyxy2xywh(targets[..., 2:])
        g = 0.5  # offset grid中心偏移
        # overlap offsets 按grid区域换算偏移区域,附近的4个网格上下左右
        off = torch.tensor([[1, 0], [0, 1],
                            [-1, 0], [0, -1]],
                           device=device).float()
        # anchor tensor, same as .repeat_interleave(nt)
        at = torch.arange(na).view(na, 1).repeat(1, nt)
        for idx, (mask, down_ratio) in enumerate(zip(self.anchor_masks, self.down_ratios)):

            anchors = np.array(self.anchors, dtype=np.float32)[mask] / down_ratio  # Scale
            # for i in range(len(self.anchor_masks)):
            #     anchors = self.anchors[i]
            gain[2:] = torch.tensor(pred[idx].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / torch.from_numpy(anchors[:, None]).to(device)  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                # j,k 为小于0.5的偏移 ,l,m为大于0.5的偏移
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                # t 原始target, t[j] x<.5 偏移的target, t[k] y<.5 偏移的target,
                # t[l] x>.5 偏移的target, t[m] y>.5 偏移的target
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0),\
                       torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                # z 原始target,x<0.5 +0.5 ,y<0.5 +0.5,x>.5 -0.5,y>.5 -0.5
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1],
                                        z[l] + off[2], z[m] + off[3]), 0) * g

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

    def get_yolov4_target(self,
                          pred,
                          img_metas,
                          batch_size,
                          gt_bbox,
                          gt_class,
                          gt_score):

        device = pred[0].device
        h, w = img_metas[0]['img_shape'][:2]
        tcls, tbox, indices, ignore_mask, anch = [], [], [], [], []
        ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
        lt = torch.cuda.LongTensor if pred[0].is_cuda else torch.Tensor
        # 循环处理每一层.
        for index, (mask, down_ratio) in enumerate(zip(self.anchor_masks, self.down_ratios)):
            b, a, gj, gi, gxywh = lt([]).to(device), lt([]).to(device), \
                                  lt([]).to(device), lt([]).to(device), ft([]).to(device)
            cls = lt([]).to(device)
            # 负责当前层预测的anchor变换到特征图尺度
            anchors = np.array(self.anchors, dtype=np.float32)[mask] / down_ratio
            # batch_ignore_mask.shape = [batch, num_masks, feat_h, feat_w, 1]
            batch_ignore_mask = torch.ones((batch_size, len(mask),
                                            int(h / down_ratio),
                                            int(w / down_ratio), 1)).to(device)
            # 循环处理每一张图片
            for bs in range(batch_size):

                if isinstance(gt_bbox[bs], torch.Tensor):
                    xywh = xyxy2xywh(gt_bbox[bs])
                else:
                    gt_bbox[bs] = torch.from_numpy(gt_bbox[bs]).to(device)
                    xywh = xyxy2xywh(gt_bbox[bs])

                # 如果目标太小, 直接忽略, 不处理.
                if len(xywh) == 0:
                    continue

                grid_h, grid_w = int(h / down_ratio), int(w / down_ratio)
                all_anchors_grid = np.array(self.anchors, dtype=np.float32) / down_ratio  # Scale

                ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
                ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
                ref_anchors = torch.from_numpy(ref_anchors)  # [0,0,anchor_w,anchor_h], 忽略xy位置.

                # gt变换到特征图尺度
                gt = xywh * torch.tensor(([grid_w, grid_h, grid_w, grid_h])).to(device).float()
                score, _cls = gt_score[bs], gt_class[bs]
                # cx_grid, cy_grid表示gt所在网格的位置.
                cx_grid = gt[:, 0].floor().cpu().numpy()  # grid_x grid_y
                cy_grid = gt[:, 1].floor().cpu().numpy()  # grid_y
                n = len(gt)
                truth_box = torch.zeros(n, 4)
                truth_box[:n, 2:4] = gt[:n, 2:4]  # gt
                anchor_ious = box_iou(truth_box, ref_anchors)  # gt与anchor算iou

                # 返回按行比较最大值的位置
                best_n_all = anchor_ious.argmax(dim=1)  # 基于max_iou, 每个gt对应的最佳anchor
                best_n = best_n_all % 3
                # 查看是否和当前尺度有最大值得IOU交集,如果有为1,否则为0
                best_n_mask = ((best_n_all == mask[0])
                               | (best_n_all == mask[1])
                               | (best_n_all == mask[2]))

                # 如果发现gt与该层负责的anchor不是匹配最佳, 则直接返回, 否则处理那些由该层负责的gt
                if sum(best_n_mask) == 0:
                    continue

                # cx,cy包含位置和偏移量,整数位代表坐标位置,小数位代表偏移量
                truth_box[:n, 0:2] = gt[:n, 0:2]
                single_ignore_mask = np.zeros((len(mask), grid_h, grid_w, 1), dtype=np.float32)
                # truth框和anchor对应的预测框的IOU,含位置信息, pred_ious.shape=[all_anchors, gt]
                pred_ious = box_iou(pred[index][bs, ..., :4].reshape(-1, 4),
                                    truth_box.reshape(-1, 4).to(device), xyxy=False)
                # 每个anchor对应的pred_bbox与所有gt的iou的最大值.
                pred_best_iou, _ = pred_ious.max(dim=1)  # [最大值,索引]
                # 如果大于ignore_thre, 则该anchor为忽略样本, False为负样本.True为忽略样本.
                pred_best_iou = (pred_best_iou > self.ignore_thre)
                # 映射到具体位置, 同理True为忽略样本, False为负样本.
                pred_best_iou = pred_best_iou.view(single_ignore_mask.shape)
                # set mask to zero (ignore) if pred matches truth
                # 取反,single_ignore_mask中True表示负样本, False表示忽略样本.
                single_ignore_mask = ~ pred_best_iou

                # torch.ones(len(truth_box))[best_n_mask].to(device)
                b = torch.cat((b, torch.ones(len(truth_box))[best_n_mask].long().to(device) * bs))
                a = torch.cat((a, best_n[best_n_mask].to(device).long()))
                gi = torch.cat((gi, torch.from_numpy(cx_grid)[best_n_mask].to(device).long()))
                gj = torch.cat((gj, torch.from_numpy(cy_grid)[best_n_mask].to(device).long()))
                gxywh = torch.cat((gxywh, truth_box[best_n_mask].to(device)))
                cls = torch.cat((cls, torch.from_numpy(_cls)[best_n_mask].to(device).long()))
                single_ignore_mask[a, gj, gi] = 0  # 正样本位置变为False, single_ignore_mask为True的地方为负样本
                # ignore_mask[gj, gi, a] = 0
                batch_ignore_mask[bs, :] = single_ignore_mask

            indices.append((b, a, gj, gi))  # indices为正样本下标.
            gxywh[..., :2] = gxywh[..., :2] - gxywh[..., :2].long()
            tbox.append(gxywh)  # 正样本回归target
            tcls.append(cls)    # 正样本分类target
            anch.append(anchors[a.cpu().numpy()])  # gt对应的anchor.
            ignore_mask.append(batch_ignore_mask)  # 装着负样本.

        return indices, tbox, tcls, anch, ignore_mask

    def init_weights(self):
        initialize_weights(self)