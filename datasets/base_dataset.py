'''
Author: JosieHong
Date: 2021-01-30 19:47:07
LastEditAuthor: JosieHong
LastEditTime: 2021-01-30 20:32:24
'''
import torch.utils.data as data

class Base_Dataset(data.Dataset):
    def __init__(self):
        super(Base_Dataset, self).__init__()

    def __len__(self):
        return 0

    def name(self):
        return 'BaseDataset'