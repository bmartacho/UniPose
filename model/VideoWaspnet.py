import torch
import torch.nn as nn
import torch.nn.functional as F
from model.WASPpose.waspVideo import build_wasp
from model.WASPpose.decoder import build_decoder
from model.WASPpose.backbone import build_backbone


class LSTM_0(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding):
        super(LSTM_0, self).__init__()
        self.conv_g_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_i_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_o_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        g = torch.tanh(self.conv_g_lstm(x))
        i = torch.sigmoid(self.conv_i_lstm(x))
        o = torch.sigmoid(self.conv_o_lstm(x))

        cell = torch.tanh(g*i)
        hide = o*cell

        return cell, hide


class LSTM(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding):
        super(LSTM, self).__init__()
        self.conv_gx_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_ix_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_ox_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_fx_lstm = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding)

        self.conv_gh_lstm = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_ih_lstm = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_oh_lstm = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding)
        self.conv_fh_lstm = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding)

    def forward(self, x, prevHide, prevCell):
        gx    = self.conv_gx_lstm(x)
        gh    = self.conv_gh_lstm(prevHide)
        g_sum = gx + gh
        gt    = torch.tanh(g_sum)

        ox    = self.conv_ox_lstm(x)
        oh    = self.conv_oh_lstm(prevHide)
        o_sum = ox + oh
        ot    = torch.sigmoid(o_sum)

        ix    = self.conv_ix_lstm(x)
        ih    = self.conv_ih_lstm(prevHide)
        i_sum = ix + ih
        it    = torch.sigmoid(i_sum)

        fx    = self.conv_fx_lstm(x)
        fh    = self.conv_fh_lstm(prevHide)
        f_sum = fx + fh
        ft    = torch.sigmoid(f_sum)

        cell = ft*prevCell + it*gt
        hide = ot*torch.tanh(cell)

        return cell, hide


class waspnet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(waspnet, self).__init__()
        self.stride = stride

        BatchNorm = nn.BatchNorm2d

        self.pool_center   = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

        self.backbone      = build_backbone(backbone, output_stride, BatchNorm)
        self.wasp          = build_wasp(backbone, output_stride, BatchNorm)
        self.decoder       = build_decoder('PennAction', num_classes, backbone, BatchNorm)

        self.lstm_0        = LSTM_0(39, 39, 3, 1)
        self.lstm          = LSTM(39, 39, 3, 1)

        # Middle CNN
        self.conv1 = nn.Conv2d( 39, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(128,  19, kernel_size=1, padding=0)


        if freeze_bn:
            self.freeze_bn()


    def forward(self, input, centermap, iter, previous, previousHide, previousCell):

        prev        = torch.zeros(1, 19, input.shape[3], input.shape[4]).cuda()
        prev[0]     = previous
        prevHide    = torch.zeros(1, 39, input.shape[3], input.shape[4]).cuda()
        prevHide[0] = previousHide
        prevCell    = torch.zeros(1, 39, input.shape[3], input.shape[4]).cuda()
        prevCell[0] = previousCell

        if iter == 0:
            x, low_level_feat = self.backbone(input[:,iter,:,:,:])
            x = self.wasp(x)
            x = self.decoder(x, low_level_feat)
            heatmap = F.interpolate(x, size=(centermap[:,iter,:,:,:].size()[2:]), mode='bilinear', align_corners=True)

            cell = torch.zeros(1, 39, input.shape[3], input.shape[4])
            hide = torch.zeros(1, 39, input.shape[3], input.shape[4])


        elif iter == 1:
            x, low_level_feat = self.backbone(input[:,iter,:,:,:])
            x = self.wasp(x)
            x = self.decoder(x, low_level_feat)
            features = F.interpolate(x, size=(centermap[:,0,:,:,:].size()[2:]), mode='bilinear', align_corners=True)

            concatenated = torch.cat((prev, features, centermap[:,iter,:,:,:]), dim=1)

            cell, hide = self.lstm_0(concatenated)

            heatmap = F.relu(self.conv1(hide))
            heatmap = F.relu(self.conv2(heatmap))
            heatmap = F.relu(self.conv3(heatmap))
            heatmap = F.relu(self.conv4(heatmap))
            heatmap = F.relu(self.conv5(heatmap))


        else:
            x, low_level_feat = self.backbone(input[:,iter,:,:,:])
            x = self.wasp(x)
            x = self.decoder(x, low_level_feat)
            features = F.interpolate(x, size=(centermap[:,0,:,:,:].size()[2:]), mode='bilinear', align_corners=True)

            concatenated = torch.cat((prev, features, centermap[:,iter,:,:,:]), dim=1)

            cell, hide = self.lstm(concatenated, prevHide, prevCell)

            heatmap = F.relu(self.conv1(hide))
            heatmap = F.relu(self.conv2(heatmap))
            heatmap = F.relu(self.conv3(heatmap))
            heatmap = F.relu(self.conv4(heatmap))
            heatmap = F.relu(self.conv5(heatmap))

        return heatmap, cell, hide


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


