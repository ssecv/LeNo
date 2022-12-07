import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet50
from multi_scale_module import FoldConv_aspp

################################ResNet#######################################
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=dilation,padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LeNo_BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,factor=0.25):
        super(LeNo_BottleNeck, self).__init__()

        if planes == 16:
            self.sigma_map = nn.Parameter(torch.ones((64, 178, 356)) * factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((64, 356, 178)) * factor, requires_grad=True)
        elif planes == 32:
            self.sigma_map = nn.Parameter(torch.ones((128, 88, 176)) * factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((128, 176, 88)) * factor, requires_grad=True)
        if planes == 64:
            self.sigma_map = nn.Parameter(torch.ones((256, 48, 96)) * factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((256, 96, 48)) * factor, requires_grad=True)
        elif planes == 128:
            self.sigma_map = nn.Parameter(torch.ones((512, 24, 48)) *factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((512, 48, 24)) *factor, requires_grad=True)
        elif planes == 256 :
            self.sigma_map = nn.Parameter(torch.ones((1024, 12, 24)) * factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((1024, 24, 12)) * factor, requires_grad=True)
        else :
            self.sigma_map = nn.Parameter(torch.ones((2048, 12, 24)) * factor, requires_grad=True)
            self.sigma_map_2 = nn.Parameter(torch.ones((2048, 24, 12)) * factor, requires_grad=True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        h_w = int(self.normal_noise.shape[1] / 2)

        self.normal_noise = torch.ones_like(self.sigma_map).normal_(0, 1).cuda()
        self.perf = self.normal_noise * self.sigma_map
        self.perf = F.pad(self.perf, pad=[0, 0, h_w, h_w], mode='constant')
        self.final_noise = self.perf.expand(out.size())
        out += self.final_noise

        self.normal_noise_2 = torch.ones_like(self.sigma_map_2).normal_(0, 1).cuda()
        self.perf_2 = self.normal_noise_2 * self.sigma_map_2
        self.perf_2 = F.pad(self.perf_2, pad=[h_w, h_w, 0, 0],mode='constant')
        self.final_noise_2 = self.perf_2.expand(out.size())
        out += self.final_noise_2

        return self.relu(out)

class GateNet(nn.Module):
    def __init__(self,block1=Bottleneck,layers=[3,4,6,3]):
        super(GateNet, self).__init__()

        ##############ResNet50---keep the last resolution###################
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block1, 64, layers[0])
        self.layer2 = self._make_layer(block1, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block1, 256, layers[2], stride=2)
        ##############change the origional stride=2 to stride=1 aim to keep the resolution for fitting the FoldConv_aspp###########
        self.layer4 = self._make_layer(block1, 512, layers[3], stride=1)
        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(64+64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(256+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(512+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(1024+256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                                nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(2048+512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                                 nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                 nn.Conv2d(128, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        self.dem1 = nn.Sequential(FoldConv_aspp(in_channel=2048,out_channel=512,out_size=384 // 16,kernel_size=3,stride=1,padding=2,dilation=2,win_size=2,win_padding=0,), nn.BatchNorm2d(512), nn.PReLU())
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.out_res = nn.Sequential(nn.Conv2d(512+256+128+64+64+1, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

        self.upsample2= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # cross-shaped noise
        self.sigma_map=nn.Parameter(torch.ones((64, 96, 192)) * 0.25, requires_grad=True)
        self.sigma_map_2 = nn.Parameter(torch.ones((64, 192, 96)) * 0.25, requires_grad=True)
        self.normal_noise = torch.ones_like(self.sigma_map).normal_(0,1).cuda()
        self.normal_noise_2 = torch.ones_like(self.sigma_map_2).normal_(0,1).cuda()

        # normal shaped noise
        # self.sigma_map=nn.Parameter(torch.ones((64, 96, 96)) * 0.25, requires_grad=True)
        # self.normal_noise = torch.ones_like(self.sigma_map).normal_(0,1).cuda()
        # self.normal_noise/=2.0
        # self.normal_noise_2 /=2.0

    def forward(self, x):
        input = x
        B,_,_,_ = input.size()
        ################################Encoder block#######################################
        x = self.conv1(x)
        x = self.bn1(x)

        # cross-shaped noise
        h_w = int(self.normal_noise.shape[1] / 2)  # 256,44,44
        self.perf = self.normal_noise * self.sigma_map  # normal_noise:cpu   sigma_map:cuda:0
        self.perf = F.pad(self.perf, pad=[0, 0, h_w, h_w], mode='constant')
        self.final_noise = self.perf.expand(x.size())
        x += self.final_noise
        self.perf_2 = self.normal_noise_2 * self.sigma_map_2
        self.perf_2 = F.pad(self.perf_2, pad=[h_w, h_w, 0, 0], mode='constant')
        self.final_noise_2 = self.perf_2.expand(x.size())
        x += self.final_noise_2

        # normal shaped noise
        # self.perf = self.normal_noise * self.sigma_map  # normal_noise:cpu   sigma_map:cuda:0
        # self.perf = F.pad(self.perf, pad=[h_w, h_w, h_w, h_w], mode='constant')
        # self.final_noise = self.perf.expand(x.size())
        # x += self.final_noise

        E1 = self.relu(x)
        x = self.maxpool(E1)

        noise_gt = self.upsample2(torch.unsqueeze(torch.mean(x, dim=1),1))

        E2 = self.layer1(x) # 8,256,96,96
        E3 = self.layer2(E2) # 8,512,48,48
        E4 = self.layer3(E3)
        E5 = self.layer4(E4)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((E5, T5), 1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5), 1)
        D5 = self.output1(G5[:, 0, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * T5)

        G4 = self.attention_feature2(torch.cat((E4,F.upsample(D5, size=E4.size()[2:], mode='bilinear')),1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4),1)
        D4 = self.output2(F.upsample(D5, size=E4.size()[2:], mode='bilinear')+G4[:, 0,:,:].unsqueeze(1).repeat(1,256,1,1)*T4)

        G3 = self.attention_feature3(torch.cat((E3,F.upsample(D4, size=E3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear')+G3[:, 0,:,:].unsqueeze(1).repeat(1,128,1,1)*T3) # 8,64,,48,48

        G2 = self.attention_feature4(torch.cat((E2,F.upsample(D3, size=E2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        D2 = self.output4(F.upsample(D3, size=E2.size()[2:], mode='bilinear')+G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T2)

        G1 = self.attention_feature5(torch.cat((E1,F.upsample(D2, size=E1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)

        tmp=F.upsample(D2, size=E1.size()[2:], mode='bilinear')+G1[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T1 # 8.64,192,192

        side=[]
        side1= self.upsample2(torch.unsqueeze(tmp[:, 11, :, :], 1)) # 11  0
        side2= self.upsample2(torch.unsqueeze(tmp[:, 42, :, :], 1)) # 42 1
        side3= self.upsample2(torch.unsqueeze(tmp[:, 57, :, :], 1)) # 57 4
        side.append(side1)
        side.append(side2)
        side.append(side3)
        noise_pre= torch.unsqueeze(tmp[:, 0, :, :], 1)

        D1 = self.output5(tmp)
        ################################ Gated Parallel & Dual branch residual fuse #######################################
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        output_res = self.out_res(torch.cat((D1,F.upsample(G5[:, 1,:,:].unsqueeze(1).repeat(1,512,1,1)*T5,size=E1.size()[2:], mode='bilinear'),F.upsample(G4[:, 1,:,:].unsqueeze(1).repeat(1,256,1,1)*T4,size=E1.size()[2:], mode='bilinear'),F.upsample(G3[:, 1,:,:].unsqueeze(1).repeat(1,128,1,1)*T3,size=E1.size()[2:], mode='bilinear'),F.upsample(G2[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T2,size=E1.size()[2:], mode='bilinear'),F.upsample(G1[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T1,size=E1.size()[2:], mode='bilinear')),1))
        output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn + output_res

        #######################################################################
        if self.training:
            # for training phase 1
            return output_fpn, pre_sal
            # for training phase 2
            # return output_fpn, output_final, noise_gt, noise_pre, side
        return pre_sal


    def _make_layer(self, block, planes, blocks, stride=1,dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def _make_layer_LeNo(self, block, planes, blocks, stride=1,dilation=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride,bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks-1):
    #         layers.append(block(self.inplanes, planes))
    #
    #     layers.append(LeNo_BottleNeck(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)



