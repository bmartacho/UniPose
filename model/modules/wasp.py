import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _AtrousModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class wasp(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(wasp, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            #dilations = [ 6, 12, 18, 24]
            dilations = [24, 18, 12,  6]
            #dilations = [6, 6, 6, 6]
        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.aspp1 = _AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        # self.global_avg_pool = nn.Sequential(nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(256,256,1,bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)
        
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_wasp(backbone, output_stride, BatchNorm):
    return wasp(backbone, output_stride, BatchNorm)
