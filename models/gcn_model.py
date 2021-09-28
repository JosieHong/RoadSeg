import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

from .gcn_parts import GCN, BR

class FCN_GCN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, input_size=(512, 640)): 
        super(FCN_GCN, self).__init__()

        self.n_channels = n_channels # 3
        self.n_classes = n_classes # 21 in paper
        self.bilinear = bilinear
        
        resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 # BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256, self.n_classes, 55) #gcn_i after layer-1
        self.gcn2 = GCN(512, self.n_classes, 27)
        self.gcn3 = GCN(1024, self.n_classes, 13)
        self.gcn4 = GCN(2048, self.n_classes, 7)

        self.br1 = BR(n_classes)
        self.br2 = BR(n_classes)
        self.br3 = BR(n_classes)
        self.br4 = BR(n_classes)
        self.br5 = BR(n_classes)
        self.br6 = BR(n_classes)
        self.br7 = BR(n_classes)
        self.br8 = BR(n_classes)
        self.br9 = BR(n_classes)

        if self.bilinear:
            strides = [16, 8, 4, 4, 1]
            sizes = [(input_size[0]//s, input_size[1]//s) for s in strides]
            self.up1 = nn.Upsample(size=sizes[0], mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(size=sizes[1], mode='bilinear', align_corners=True)
            self.up3 = nn.Upsample(size=sizes[2], mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(size=sizes[3], mode='bilinear', align_corners=True)
            
            self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up6 = nn.Upsample(size=sizes[4], mode='bilinear', align_corners=True)
        else:
            print("Error: please set bilinear=True in FCN-GCN")
            exit()

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, self.n_channels, padding=1, bias=False),
            nn.BatchNorm2d(in_c/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(in_c/2, self.n_classes, 1))    

    def forward(self,x):
        # input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        convA_x = x
        x = self.maxpool(x)
        pooled_x = x

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        # print(fm3.size()[2:], fm2.size()[2:], fm1.size()[2:], pooled_x.size()[2:], input.size()[2:])
        # torch.Size([32, 40]) torch.Size([64, 80]) torch.Size([128, 160]) torch.Size([128, 160]) torch.Size([512, 640])

        gc_fm4 = self.up1(gc_fm4)
        gc_fm3 = self.up2(self.br5(gc_fm3 + gc_fm4))
        gc_fm2 = self.up3(self.br6(gc_fm2 + gc_fm3))
        gc_fm1 = self.up4(self.br7(gc_fm1 + gc_fm2))

        gc_fm1 = self.up5(self.br8(gc_fm1))
        out = self.up6(self.br9(gc_fm1))

        # gc_fm4 = F.Upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        # gc_fm3 = F.Upsample(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        # gc_fm2 = F.Upsample(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        # gc_fm1 = F.Upsample(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        # gc_fm1 = F.Upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)
        # out = F.Upsample(self.br9(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)
        return out