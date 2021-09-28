import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dice_loss import dice_coeff

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

def test_net(net, loader, device):
    net.eval()
    n_val = len(loader)  # the number of batch
    metrics = {}
    if net.n_classes > 1:
        for i in range(1, net.n_classes+1): # 'road', 'car', 'others'
            metrics[str(i)+'_IoU'] = np.array([])
            metrics[str(i)+'_ACC'] = np.array([])
    else:
        for i in range(net.n_classes+1): # 'bg', 'road'
            metrics[str(i)+'_IoU'] = np.array([])
            metrics[str(i)+'_ACC'] = np.array([])

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, gt = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            gt = gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                pred = pred.transpose(0, 1)
                gt = gt.transpose(0, 1)
                for i, (p, g) in enumerate(zip(pred, gt)):
                    if torch.sum(g) == 0:
                        continue
                    iou = torch.sum(torch.logical_and(p, g)) / torch.sum(torch.logical_or(p, g))
                    acc = torch.sum(torch.logical_and(p, g)) / torch.sum(g)
                    metrics[str(i+1)+'_IoU'] = np.append(metrics[str(i+1)+'_IoU'], iou.cpu())
                    metrics[str(i+1)+'_ACC'] = np.append(metrics[str(i+1)+'_ACC'], acc.cpu())
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                bg_gt = 1 - gt
                bg_pred = 1 - pred

                iou0 = torch.sum(torch.logical_and(bg_pred, bg_gt)) / torch.sum(torch.logical_or(bg_pred, bg_gt))
                acc0 = torch.sum(torch.logical_and(bg_pred, bg_gt)) / torch.sum(bg_gt)
                metrics['0_IoU'] = np.append(metrics['0_IoU'], iou0.cpu())
                metrics['0_ACC'] = np.append(metrics['0_ACC'], acc0.cpu())

                iou1 = torch.sum(torch.logical_and(pred, gt)) / torch.sum(torch.logical_or(pred, gt))
                acc1 = torch.sum(torch.logical_and(pred, gt)) / torch.sum(gt)
                metrics['1_IoU'] = np.append(metrics['1_IoU'], iou1.cpu())
                metrics['1_ACC'] = np.append(metrics['1_ACC'], acc1.cpu())

            pbar.update()

    # print(metrics)
    for k, v in metrics.items():
        metrics[k] = v.mean()
    return metrics
