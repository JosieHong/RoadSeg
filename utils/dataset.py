import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, root_dir, scale=1):
        self.classes = []
        self.root_dir = root_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = []

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, nol=True):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1 and nol:
            img_trans = img_trans / img_trans.max()

        return img_trans

    def __getitem__(self, i):
        '''Please impelement it in the costum dataset'''
        pass


class TSDDataset_bin(BasicDataset):
    def __init__(self, root_dir, list_dir, scale): 
        super().__init__(root_dir, scale)

        self.classes = ['road']
        self.root_dir = root_dir
        with open(list_dir, 'r') as f:
            self.ids = f.read().splitlines()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, i): 
        img_path = os.path.join(self.root_dir, self.ids[i].split(" ")[0])
        label_path = os.path.join(self.root_dir, self.ids[i].split(" ")[1])

        img = Image.open(img_path)
        mask = Image.open(label_path)
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale, nol=False)

        new_mask = np.zeros(mask.shape)
        new_mask[mask==1] = 1

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(new_mask).type(torch.FloatTensor)
        }

    
class TSDDataset_mul(BasicDataset):
    def __init__(self, root_dir, list_dir, scale): 
        super().__init__(root_dir, scale)

        self.classes = ['road', 'car', 'others']
        self.root_dir = root_dir
        with open(list_dir, 'r') as f:
            self.ids = f.read().splitlines()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # for loss weights 
        self.classes_prob = self.get_class_prob(self.ids)
        self.img_size = (int(1024*scale), int(1280*scale))

    def __getitem__(self, i): 
        img_path = os.path.join(self.root_dir, self.ids[i].split(" ")[0])
        label_path = os.path.join(self.root_dir, self.ids[i].split(" ")[1])

        img = Image.open(img_path)
        mask = Image.open(label_path)
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale, nol=False)

        mask = mask[0] # (1, H, W) -> (H, W)
        H, W = mask.shape
        new_mask = np.zeros((len(self.classes), H, W))
        new_mask[0][mask==1] = 1
        new_mask[1][mask==2] = 1
        new_mask[2][mask==3] = 1

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(new_mask).type(torch.FloatTensor)
        }

    def get_class_prob(self, file_names):
        count = np.zeros((len(self.classes)))
        label_paths = [os.path.join(self.root_dir, i.split(" ")[1]) for i in file_names]
        for label_path in label_paths:
            mask = np.asarray(Image.open(label_path))
            for i in range(len(self.classes)):
                count[i] += np.sum(mask==i)
        return [c/np.sum(count) for c in count]