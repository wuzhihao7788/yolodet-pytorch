#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/8/10 19:11
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: test.py 
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
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from yolodet.models.heads.base import non_max_suppression, clip_coords, scale_coords, xyxy2xywh
from yolodet.models.loss.base import box_iou
from yolodet.models.utils import torch_utils
from yolodet.utils.util import ap_per_class, coco80_to_coco91_class


def single_gpu_test(model, data_loader,half=False,conf_thres=0.001,iou_thres = 0.6,merge = False,save_json=False,augment=False,verbose=False,coco_val_path =''):

    device = next(model.parameters()).device  # get model device
    # Half
    half = device.type != 'cpu' and half  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5
    niou = iouv.numel()

    seen = 0
    nc = model.head.num_classes
    names = model.CLASSES if hasattr(model, 'CLASSES') else data_loader.dataset.CLASSES
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, batch in enumerate(tqdm(data_loader, desc=s)):

        img = batch['img'].to(device, non_blocking=True)
        batch['img'] = img.half() if half else img.float()  # uint8 to fp16/32
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        ft = torch.cuda.FloatTensor if half else torch.Tensor
        gt_bbox = batch['gt_bboxes']
        gt_class = batch['gt_class']
        img_metas = batch['img_metas']

        targets = ft([]).to(device)
        for i,gtb in enumerate(gt_bbox):
            gtc = torch.from_numpy(gt_class[i]).to(device)
            img_idx = torch.ones(len(gtb),1,device=device)*i
            targets = torch.cat([targets,torch.cat((img_idx,gtc,torch.from_numpy(gtb).to(device)),dim=-1)])

            # Disable gradients
        with torch.no_grad():
            # Run model
            batch['eval'] = True
            if augment:
                batch['augment'] = True
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(return_loss=False, **batch)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # if save_txt:
            #     filename = img_metas[si]['filename']
            #     ori_shape = img_metas[si]['ori_shape']
            #     # img_shape = img_metas[si]['img_shape']
            #
            #     # gn = torch.tensor(ori_shape[:2])[[0, 1, 0, 1]]  # normalization gain whwh
            #     txt_path = str(out / Path(filename).stem)
            #     pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], ori_shape[:2])  # to original
            #     for *xyxy, conf, cls in pred:
            #         # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         with open(txt_path + '.txt', 'a') as f:
            #             f.write(('%g ' * 5 + '\n') % (cls, *xyxy))  # label format

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            # if save:
            #     _pd = pred.cpu().numpy()
            #     for _p in _pd:
            #         left_top = (int(_p[0]), int(_p[1]))
            #         right_bottom = (int(_p[2]), int(_p[3]))
            #         cv2.rectangle(
            #             img, left_top, right_bottom, color=(0, 0, 255), thickness=2)
            #         label_text = str(_p[5])
            #         label_text += '|{:.02f}'.format(_p[4])
            #         cv2.putText(img, label_text, (int(_p[0]), int(_p[1]) - 2), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,color=(0, 0, 255))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                filename = img_metas[si]['filename']
                ori_shape = img_metas[si]['ori_shape']
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, ori_shape[:2])  # to original shape
                image_id = str(Path(filename).stem)
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                tbox = labels[:, 1:5] * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # Plot images
            # if batch_i < 1:
            #     f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            #     plot_images(img, targets, paths, str(f), names)  # ground truth
            #     f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            #     plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions

        # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (height, width, data_loader.batch_size)  # tuple

    # Save JSON
    if save_json and len(jdict):
        filename = model.cfg.filename
        basename = os.path.basename(filename)
        bname = os.path.splitext(basename)[0]
        f = 'detections_val2017_%s_results.json' % bname  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        print('\nCOCO mAP with pycocotools... saving %s finished' % f)
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in data_loader.dataset.imgs]
            cocoGt = COCO(
                glob.glob(coco_val_path+'/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(data_loader)).tolist()), maps, t


