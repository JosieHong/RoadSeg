'''
Author: JosieHong
Date: 2021-01-30 16:11:05
LastEditAuthor: JosieHong
LastEditTime: 2021-02-21 18:01:54
'''
import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from .base_dataset import Base_Dataset

class TSD_Dataset(Base_Dataset):
    """dataloader for kitti dataset"""
    def __init__(self, root, mode, no_label, img_size):
        super(TSD_Dataset, self).__init__()
        self.root = root # path to the dataset
        self.mode = mode
        self.no_label = no_label
        self.num_labels = 2
        self.use_size = (img_size[0], img_size[1])
        
        # split the training data into 'training', 'validation' and 'testing'
        if self.mode == "train":
            with open(os.path.join(self.root, "train_lst.txt")) as f:
                image_list = f.read().splitlines()
            length = len(image_list)
            self.image_list = image_list[:int(length*0.8)]
        elif self.mode == "val":
            with open(os.path.join(self.root, "train_lst.txt")) as f:
                image_list = f.read().splitlines()
            length = len(image_list)
            self.image_list = image_list[:int(length*0.8)]
        else:
            with open(os.path.join(self.root, "test_lst.txt")) as f:
                self.image_list = f.read().splitlines()

    def __getitem__(self, index):
        img_path = self.image_list[index].split(" ")[0]
        label_path = self.image_list[index].split(" ")[1]
        
        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(self.root, img_path)), cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = rgb_image.shape
        if self.mode == 'test' and self.no_label:
            # Since we have no gt label (e.g., kitti submission), we generate pseudo gt labels to
            # avoid destroying the code architecture
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        else:
            label_image = cv2.cvtColor(cv2.imread(os.path.join(self.root, label_path)), cv2.COLOR_BGR2RGB)
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[label_image[:,:,2] > 0] = 1

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        rgb_image = rgb_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image)

        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)
        label[label > 0] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        
        # return a dictionary containing useful information
        # 'rgb image' and 'label' for training
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'label': label, 'path': img_path, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'tsd_max'