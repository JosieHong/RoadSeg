'''
Author: JosieHong
Date: 2021-01-30 20:36:09
LastEditAuthor: JosieHong
LastEditTime: 2021-02-01 16:35:46
'''
import torch.nn as nn
from torchvision import models

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Hall_Module(nn.Module):
    def __init__(self):
        super(Hall_Module, self).__init__()
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)

        # 'same' padding = (kernel-1)/2 = (3-1)/2 = 1
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=3, padding=(1,1)) # 'same' padding
        self.bn2 = nn.BatchNorm2d(2048)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(2048, 2048, kernel_size=3, padding=(1,1)) # 'same' padding
        self.bn3 = nn.BatchNorm2d(2048)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        return self.relu3(self.bn3(self.conv3(x1 + x2)))


class Fusion_Module(nn.Module): 
    def __init__(self, in_channel, out_channel, t_stride=(2,2), t_kernel=4, t_padding=(1,1)):
        super(Fusion_Module, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channel, out_channel, stride=t_stride, kernel_size=t_kernel, padding=t_padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=(1,1)) # 'same' padding
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu1(self.bn1(self.trans_conv(x1)))
        return self.relu2(self.bn2(self.conv(x1 + x2)))


class Kitti_Seg(nn.Module):
    def __init__(self):
        super(Kitti_Seg, self).__init__()
        self.encoder = models.resnet101(pretrained=True)
        self.neck = Hall_Module()
        self.fusion1 = Fusion_Module(2048, 1024)
        self.fusion2 = Fusion_Module(1024, 512)
        self.fusion3 = Fusion_Module(512, 256)
        self.fusion4 = Fusion_Module(256, 64, t_stride=(1,1), t_kernel=3, t_padding=(1, 1))
        self.trans_conv = nn.ConvTranspose2d(64, 1, stride=(4,4), kernel_size=16, padding=(6,6)) # Yuhui: not sure about the parameter of this layer
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        img_size = (int(x.size()[2]), int(x.size()[3]))
        # -------------------------------------
        # Encoder (ResNet101)
        # 
        # output features' size:
        #   torch.Size([batch_size, 64, 64, 64]) 
        #   torch.Size([batch_size, 256, 64, 64]) 
        #   torch.Size([batch_size, 512, 32, 32]) 
        #   torch.Size([batch_size, 1024, 16, 16]) 
        #   torch.Size([batch_size, 2048, 8, 8])
        # -------------------------------------
        output = self.encoder.conv1(x)
        output = self.encoder.bn1(output)
        output = self.encoder.relu(output)
        feature1 = self.encoder.maxpool(output)
        feature2 = self.encoder.layer1(feature1)
        feature3 = self.encoder.layer2(feature2)
        feature4 = self.encoder.layer3(feature3)
        feature5 = self.encoder.layer4(feature4)
        
        # -------------------------------------
        # Neck (Hall_Module)
        # 
        # output features' size:
        #   torch.Size([batch_size, 2048, 8, 8])
        # -------------------------------------
        output = self.neck(feature5)
        
        # -------------------------------------
        # Decoder (FCN)
        # 
        # output features' size:
        #   torch.Size([batch_size, 2048, 8, 8])
        #   torch.Size([batch_size, 1024, 16, 16])
        #   torch.Size([batch_size, 512, 32, 32])
        #   torch.Size([batch_size, 256, 64, 64])
        #   torch.Size([batch_size, 64, 64, 64])
        # -------------------------------------
        output = self.fusion1(output, feature4)
        output = self.fusion2(output, feature3)
        output = self.fusion3(output, feature2)
        output = self.fusion4(output, feature1)
        
        # -------------------------------------
        # Last layer (ajust the shape of output)
        # 
        # output features' size:
        #   torch.Size([batch_size, 1, 256, 256])
        # -------------------------------------
        output = self.trans_conv(output)
        
        return self.relu(self.bn(output)).view(-1, img_size[0], img_size[1]) # torch.Size([batch_size, 256, 256])