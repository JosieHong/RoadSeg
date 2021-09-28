import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.eval import eval_net, test_net
from utils.dataset import TSDDataset_bin, TSDDataset_mul
from models.unet_model import UNet
from models.gcn_model import FCN_GCN
from models.our_model import Road_Seg

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

dir_root = 'data/xjtu_plus/'
# dir_img = 'data/xjtu_plus/img/'
# dir_mask = 'data/xjtu_plus/label/'
dir_list = 'data/xjtu_plus/train_lst.txt'
dir_list_test = 'data/xjtu_plus/test_lst.txt'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              dataset='TSDDataset_bin',
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    if dataset == 'TSDDataset_bin':
        dataset = TSDDataset_bin(dir_root, dir_list, img_scale)
    elif dataset == 'TSDDataset_mul': 
        dataset = TSDDataset_mul(dir_root, dir_list, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # get the class probabilities
    if net.n_classes > 1:
        # print("classes_prob:", dataset.classes_prob)
        # print("img_size:", dataset.img_size)
        # print("class_num:", len(dataset.classes))
        weights = torch.zeros((len(dataset.classes), dataset.img_size[0], dataset.img_size[1])).to(device)
        for i in range(len(dataset.classes)):
            weights[i] = torch.full(dataset.img_size, 1/dataset.classes_prob[i])
        criterion = nn.BCEWithLogitsLoss(weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

        if save_cp:
            try:
                os.mkdir(os.path.join(dir_checkpoint, args.model))
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint +args.model+'/'+ f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default='TSDDataset_bin',
                        help='Dataset type: [TSDDataset_bin/TSDDataset_mul]')
    parser.add_argument('-m', '--model', dest='model', type=str, default='unet',
                        help='Model type: [unet/gcn/ours]')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    if args.model == 'unet':
        if args.dataset == 'TSDDataset_bin':
            net = UNet(n_channels=3, n_classes=1, bilinear=True)
        elif args.dataset == 'TSDDataset_mul':
            net = UNet(n_channels=3, n_classes=3, bilinear=True)
        
    elif args.model == 'gcn':
        if args.dataset == 'TSDDataset_bin':
            net = FCN_GCN(n_channels=3, n_classes=1, bilinear=True)
        elif args.dataset == 'TSDDataset_mul':
            net = FCN_GCN(n_channels=3, n_classes=3, bilinear=True)

    elif args.model == 'ours':
        if args.dataset == 'TSDDataset_bin':
            net = Road_Seg(n_channels=3, n_classes=1, bilinear=True)
        elif args.dataset == 'TSDDataset_mul':
            net = Road_Seg(n_channels=3, n_classes=3, bilinear=True)

    else:
        print("Error: {} is not supported, please chooes a model in [unet/gcn]".format(args.model))
        exit()

    logging.info(f'Network:\n'
                f'\t{net.n_channels} input channels\n'
                f'\t{net.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                    dataset=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    try:
        if args.dataset == 'TSDDataset_bin':
            dataset = TSDDataset_bin(dir_root, dir_list_test, args.scale)
        elif args.dataset == 'TSDDataset_mul':
            dataset = TSDDataset_mul(dir_root, dir_list_test, args.scale)
        test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        metrics = test_net(net, loader=test_loader, device=device)
        for k, v in metrics.items():
            print("{}: {}".format(k, v))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)