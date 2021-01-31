'''
Author: JosieHong
Date: 2021-01-30 23:21:03
LastEditAuthor: JosieHong
LastEditTime: 2021-01-31 12:12:12
'''
import torch

def mask_iou(mask1, mask2):
    """
    masks1: [b, w, h]
    masks2: [b, w, h]
    """
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score