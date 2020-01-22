import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
from model.WASPpose.wasp     import build_wasp
from model.WASPpose.decoder  import build_decoder
from model.WASPpose.backbone import build_backbone
from model.YOLO_layer        import YOLOLayer

class waspnet(nn.Module):
    def __init__(self, dataset, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(waspnet, self).__init__()
        self.stride = stride

        BatchNorm = nn.BatchNorm2d

        self.num_classes = num_classes

        self.pool_center   = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

        self.backbone      = build_backbone(backbone, output_stride, BatchNorm)
        self.wasp          = build_wasp(backbone, output_stride, BatchNorm)
        self.decoder       = build_decoder(dataset, num_classes, backbone, BatchNorm)

        self.conv1 = nn.Conv2d(256, 128, 1, bias=False)
        self.conv2 = nn.Conv2d(1, 128, 1, bias=False)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, center_map, targets=None):

        x, low_level_feat = self.backbone(input)

        x = self.wasp(x)

        # low_level = torch.cat([self.conv1(low_level_feat), F.interpolate(self.conv2(center_map), size=(low_level_feat.size()[2:]), mode='bilinear', align_corners=True)], dim=1)
        x = self.decoder(x, low_level_feat)

        # x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = F.interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)

        return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = waspnet(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


