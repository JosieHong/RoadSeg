'''
Author: JosieHong
Date: 2021-01-30 15:59:44
LastEditAuthor: JosieHong
LastEditTime: 2021-02-01 01:05:29
'''
import argparse
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.kitti_dataset import KITTI_Dataset
from models.kitti_seg import Kitti_Seg, weight_init
from utils import mask_iou

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument(
    '--nepoch', type=int, default=12, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='kitti', help="dataset type")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'kitti':
    dataset = KITTI_Dataset(
        root=opt.dataset,
        mode='train',
        no_label = False, 
        img_size=(256,256))

    test_dataset = KITTI_Dataset(
        root=opt.dataset,
        mode='val',
        no_label = True,
        img_size=(256,256))
else:
    exit('wrong dataset type')

dataloader = DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)
testdataloader = DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

print('Load Dataset!\ntrain dataset:', len(dataset), 'test dataset:', len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

seg_model = Kitti_Seg()
seg_model.apply(weight_init)
print("Build the model!")

if opt.model != '':
    print("Load the pretrained model from {}".format(opt.model))
    seg_model.load_state_dict(torch.load(opt.model))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(seg_model.parameters(), lr=0.00001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
seg_model.cuda()

num_batch = len(dataset) / opt.batchSize

val_loss = [] # for debug
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        # {'rgb_image': rgb_image, 'label': label, 'path': name, 'oriSize': (oriWidth, oriHeight)}
        rgb_image = data['rgb_image']
        target = data['label']
        # path = data['path']
        # ori_size = data['oriSize']
        rgb_image, target = rgb_image.cuda(), target.cuda().float()
        optimizer.zero_grad()
        seg_model = seg_model.train()
        pred = seg_model(rgb_image) # torch.Size([batch_size, oriWidth, oriHeight])
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        acc = mask_iou(pred.detach(), target.detach())
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), acc))

        if i % 12 == 0:
            j, data = next(enumerate(testdataloader, 0))
            rgb_image = data['rgb_image']
            target = data['label']
            rgb_image, target = rgb_image.cuda(), target.cuda().float()
            seg_model = seg_model.eval()
            pred = seg_model(rgb_image)
            loss = criterion(pred, target)
            val_loss.append(loss.item()) # for debug
            acc = mask_iou(pred.detach(), target.detach())
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), acc))
    scheduler.step()
    torch.save(seg_model.state_dict(), '%s/model_%d.pth' % (opt.outf, epoch))
# for debug
with open(os.path.join(opt.outf, "val_loss.json"), "w") as json_file:
    json.dump(val_loss, json_file)
    print("Save the validation loss in {}".format(os.path.join(opt.outf, "val_loss.json")))

accuracy = 0
results = []
for i, data in tqdm(enumerate(testdataloader, 0)):
    rgb_image = data['rgb_image']
    target = data['label']
    rgb_image, target = rgb_image.cuda(), target.cuda().float()
    with torch.no_grad(): # save the space of GPU
        seg_model = seg_model.eval()
        pred = seg_model(rgb_image)
    acc = mask_iou(pred.detach(), target.detach())
    accuracy += acc
    results.append({"target": target.tolist(), "pred": pred.tolist()})
print("final accuracy {}".format(accuracy/len(testdataloader)))

with open(os.path.join(opt.outf, "test_results.json"), "w") as json_file:
    json.dump(results, json_file)
    print("Save the test results in {}".format(os.path.join(opt.outf, "test_results.json")))