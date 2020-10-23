import torch.nn as nn
import numpy as np

def conf_accuracy(pred, target,thre=.5):

    pred = pred >= thre  # 过滤掉大于阈值的数据，小于阈值1，大于0
    pred = pred.type_as(target)

    correct = pred.eq(target).view(-1)
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / correct.size(0))

def class_accuracy(pred, target, topk=1):
    assert isinstance(topk, int)
    _, pred_label = pred.topk(topk, dim=-1)# return max value，index
    _, target = target.topk(topk, dim=-1)# return max value，index
    pred_label = pred_label.view(-1,1)
    correct = pred_label.eq(target.view(-1,1).expand_as(pred_label))
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / pred_label.size(0))


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)
