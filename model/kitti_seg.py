'''
Author: JosieHong
Date: 2021-01-30 20:36:09
LastEditAuthor: JosieHong
LastEditTime: 2021-01-31 02:03:00
'''
import torch.nn as nn
from .resnet import resnet50


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Hall_Module(nn.Module):
    def __init__(self):
        super(Hall_Module, self).__init__()
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=3, padding=(1,1)) # padding = (kernel-1)/2 = (3-1)/2 = 1
        self.conv3 = nn.Conv2d(2048, 2048, kernel_size=3, padding=(1,1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.conv3(x1 + x2)


class Fusion_Module(nn.Module):
    def __init__(self, in_channel, out_channel, t_stride=(2,2), t_kernel=4, t_padding=(1,1)):
        super(Fusion_Module, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channel, out_channel, stride=t_stride, kernel_size=t_kernel, padding=t_padding)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=(1,1))
        
    def forward(self, x1, x2):
        x1 = self.trans_conv(x1)
        return self.conv(x1 + x2)


class Kitti_Seg(nn.Module):
    def __init__(self):
        super(Kitti_Seg, self).__init__()
        self.encoder = resnet50()
        self.neck = Hall_Module()
        self.de_conv1 = Fusion_Module(2048, 1024)
        self.de_conv2 = Fusion_Module(1024, 512)
        self.de_conv3 = Fusion_Module(512, 256)
        self.de_conv4 = Fusion_Module(256, 64, t_stride=(1,1), t_kernel=3, t_padding=(1,1)) # Yuhui: I am not sure about the parameter of this layer
        self.conv = nn.Conv2d(64, 1, kernel_size=1)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        img_size = (int(x.size()[2]), int(x.size()[3]))
        feature1, feature2, feature3, feature4, feature5 = self.encoder(x)
        # print(feature1.size(), feature2.size(), feature3.size(), feature4.size(), feature5.size())
        # torch.Size([1, 64, 256, 256])
        # torch.Size([1, 256, 256, 256])
        # torch.Size([1, 512, 128, 128])
        # torch.Size([1, 1024, 64, 64])
        # torch.Size([1, 2048, 32, 32])
        output = self.neck(feature5)
        output = self.de_conv1(output, feature4)
        output = self.de_conv2(output, feature3)
        output = self.de_conv3(output, feature2)
        output = self.de_conv4(output, feature1)
        # torch.Size([1, 2048, 32, 32])
        # torch.Size([1, 1024, 64, 64])
        # torch.Size([1, 512, 128, 128])
        # torch.Size([1, 256, 256, 256])
        # torch.Size([1, 64, 256, 256])
        output = self.soft_max(self.conv(output)).view(-1, img_size[0], img_size[1])
        return output # torch.Size([1, 256, 256])