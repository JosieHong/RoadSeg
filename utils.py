'''
Author: JosieHong
Date: 2021-01-30 23:21:03
LastEditAuthor: JosieHong
LastEditTime: 2021-02-01 13:49:40
'''
import torch
import numpy as np

def mask_iou(mask1, mask2):
    """
    masks1: [b, w, h]
    masks2: [b, w, h]
    """
    # If torch >= 1.5.0
    # intersection = torch.logical_and(mask1, mask2)
    # union = torch.logical_or(mask1, mask2)
    # iou_score = torch.sum(intersection) / torch.sum(union)

    # If torch < 1.5.0
    mask1 = mask1.cpu().numpy()
    mask2 = mask2.cpu().numpy()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

# Reference: https://github.com/hlwang1124/SNE-RoadSeg/blob/5e15ef16642b6b425c4074e025a1704312d06240/util/util.py
def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou