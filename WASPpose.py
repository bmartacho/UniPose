# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import cv2
import math
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import save_checkpoint      as save_checkpoint
from utils.utils import printAccuracies      as printAccuracies
from utils.utils import guassian_kernel      as guassian_kernel
from utils.utils import get_parameters       as get_parameters
from utils       import Mytransforms         as Mytransforms 
from utils.utils import getDataloader        as getDataloader
from utils.utils import getOutImages         as getOutImages
from utils.utils import AverageMeter         as AverageMeter
from utils.utils import draw_paint           as draw_paint
from utils       import evaluate             as evaluate
from utils.utils import get_kpts             as get_kpts
from utils.utils import draw_BBox            as draw_BBox
from utils.utils import non_max_suppression  as non_max_suppression

from utils.uniPose import uniPose_kpts       as uniPose_kpts
# from utils.uniPose import draw_paint           as draw_paint

from model.waspnet import waspnet

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

from PIL import Image
import scipy.misc

import warnings


class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 4
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 3
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 8

        cudnn.benchmark   = True

        if self.dataset   ==  "LSP":
            self.numClasses  = 14
        elif self.dataset == "MPII":
            self.numClasses  = 16
        elif self.dataset == "COCO":
            self.numClasses  = 17
        elif self.dataset == "NTID" or self.dataset == "NTID_small":
            self.numClasses  = 19
        elif self.dataset == "PoseTrack":
            self.numClasses  = 17
        elif self.dataset == "BBC":
            self.numClasses  = 7


        self.train_loader, self.val_loader = getDataloader(self.dataset, self.train_dir, self.val_dir, self.sigma, self.stride, self.workers, self.batch_size)

        model = waspnet(self.dataset, num_classes=self.numClasses,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)

        self.model       = model.cuda()

        self.criterion   = nn.MSELoss().cuda()

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.best_model  = 12345678.9

        self.iters       = 0

        if self.args.pretrained is not None:
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']

            if self.dataset == "LSP" or self.dataset == "NTID" or self.dataset == "NTID_small" or self.dataset == "BBC":
                prefix = 'invalid'
            elif self.dataset == "MPII" or self.dataset == "COCO" or self.dataset == "PoseTrack":
                prefix = 'decoder.last_conv.8'
            # prefix = 'decoder.last_conv.8'

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict:
                    if not k.startswith(prefix):
                        model_dict[k] = v

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)
            
        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0

        # summary(self.model,(3,960,720))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)

        for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):
        # for i, (input, heatmap) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =     input.cuda()
            heatmap_var   =    heatmap.cuda()
            limbs_var     =   limbsmap.cuda()
            box_var       =        box.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var, img_path, box)

            target = torch.cat((heatmap_var, box_var), dim=1)

            loss_heat   = self.criterion(heat,  target)

            loss = loss_heat
            train_loss += loss_heat.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1

            # if i == 10000:
            # 	break

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss    = 0.0
        yolo_losses = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK   = np.zeros(self.numClasses+1)
        PCKh  = np.zeros(self.numClasses+1)
        count = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):
        #for i, (input, heatmap) in enumerate(tbar):

            cnt += 1

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()
            limbs_var     =   limbsmap.cuda()
            box_var       =        box.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var, img_path, box)

            target = torch.cat((heatmap_var, box_var), dim=1)

            loss_heat   = self.criterion(heat,  target)

            loss = loss_heat
            val_loss += loss_heat.item()

            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(), target.detach().cpu().numpy(),0.2,0.5, self.dataset)

            AP[0]     = (AP[0]  *i + acc[0])      / (i + 1)
            PCK[0]    = (PCK[0] *i + acc_PCK[0])  / (i + 1)
            PCKh[0]   = (PCKh[0]*i + acc_PCKh[0]) / (i + 1)

            for j in range(1,self.numClasses+1):
                if visible[j] == 1:
                    AP[j]     = (AP[j]  *count[j] + acc[j])      / (count[j] + 1)
                    PCK[j]    = (PCK[j] *count[j] + acc_PCK[j])  / (count[j] + 1)
                    PCKh[j]   = (PCKh[j]*count[j] + acc_PCKh[j]) / (count[j] + 1)
                    count[j] += 1

            mAP     =   AP[1:].sum()/(self.numClasses)
            mPCK    =  PCK[1:].sum()/(self.numClasses)
            mPCKh   = PCKh[1:].sum()/(self.numClasses)


            target = F.interpolate(target, size=input_var.size()[2:], mode='bilinear', align_corners=True)
            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            # kptsT = get_kpts(target, img_h=368, img_w=368)
            # kptsH = get_kpts(heat, img_h=368, img_w=368)

            # print(kptsT)
            # print(kptsH)
            # print(pred, acc)

            # heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            # print(heat.shape)

            # kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
            # draw_paint(img_path[0], kpts, 0, epoch, self.model_arch, self.dataset)

            # im = cv2.imread(img_path[0])

            # # heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            # # kpts = uniPose_kpts(heat, self.dataset, img_h=800.0, img_w=800.0)
            # # print(kpts)



            # im  = cv2.resize(cv2.imread(img_path[0]),(368,368))

            # box_h = box[0][3]
            # box_w = box[0][2]
            # y1    = box[0][1]
            # x1    = box[0][0]

            # print(x1,y1)
            # print(x1-box_w/2, y1-box_h/2)
            # print(x1-box_w/2, y1+box_h/2)
            # print(x1+box_w/2, y1-box_h/2)
            # print(x1+box_w/2, y1+box_h/2)

            # cv2.circle(im, (y1, x1), radius=1, thickness=-1, color=(0, 0, 255))
            # cv2.circle(im, (y1-box_h/2, x1-box_w/2), radius=1, thickness=2, color=(0, 0, 255))
            # cv2.circle(im, (y1-box_h/2, x1+box_w/2), radius=1, thickness=2, color=(0, 0, 255))
            # cv2.circle(im, (y1+box_h/2, x1-box_w/2), radius=1, thickness=2, color=(0, 0, 255))
            # cv2.circle(im, (y1+box_h/2, x1+box_w/2), radius=1, thickness=2, color=(0, 0, 255))

            # [Y0, X0] = [y1-box_h/2, x1-box_w/2]
            # [Y1, X1] = [y1-box_h/2, x1+box_w/2]
            # mX = np.mean([X0, X1])
            # mY = np.mean([Y0, Y1])
            # length  = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
            # angle   = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(im, polygon, (0, 0, 255))

            # [Y0, X0] = [y1-box_h/2, x1-box_w/2]
            # [Y1, X1] = [y1+box_h/2, x1-box_w/2]
            # mX = np.mean([X0, X1])
            # mY = np.mean([Y0, Y1])
            # length  = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
            # angle   = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(im, polygon, (0, 0, 255))

            # [Y0, X0] = [y1+box_h/2, x1-box_w/2]
            # [Y1, X1] = [y1+box_h/2, x1+box_w/2]
            # mX = np.mean([X0, X1])
            # mY = np.mean([Y0, Y1])
            # length  = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
            # angle   = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(im, polygon, (0, 0, 255))

            # [Y0, X0] = [y1-box_h/2, x1+box_w/2]
            # [Y1, X1] = [y1+box_h/2, x1+box_w/2]
            # mX = np.mean([X0, X1])
            # mY = np.mean([Y0, Y1])
            # length  = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
            # angle   = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(im, polygon, (0, 0, 255))
  
            # cv2.imwrite('samples/WASPpose/BBOX_test.png', im)

            
            # if i == 2:
            #     break



        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
            
        PCKhAvg = PCKh.sum()/(self.numClasses+1)
        PCKAvg  =  PCK.sum()/(self.numClasses+1)

        if mAP > self.isBest:
            self.isBest = mAP
            save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name)
            print("Model saved to "+self.args.model_name)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK


        print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.isBest*100, self.bestPCK*100,self.bestPCKh*100))


        im = cv2.imread(img_path[0])

        if self.dataset == "BBC":
            im = im[-368:,-368:,:]


        if self.dataset == "BBC":
            kpts = get_kpts(heat, img_h=368, img_w=368)
        elif self.dataset == "NTID_small":
            kpts = get_kpts(heat, img_h=800, img_w=800)

        draw_paint(im, kpts, 0, epoch, self.model_arch, self.dataset)

        # Save one sample
        heat = heat.detach().cpu().numpy()

        heat = heat[0].transpose(1,2,0)


        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                for k in range(heat.shape[2]):
                    if heat[i,j,k] < 0:
                        heat[i,j,k] = 0
                

        im = cv2.imread(img_path[0])

        if self.dataset == "BBC":
            im = im[-368:,-368:,:]

        heatmap = []

        cv2.imwrite('samples/WASPpose/heat/WASPpose_original.png', im)

        background = -255*(heat[:,:,0]-np.amax(heat[:,:,0]))
        heatmap = cv2.applyColorMap(np.uint8(background), cv2.COLORMAP_JET)
        im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
        cv2.imwrite('samples/WASPpose/heat/WASPpose_0.png', im_heat)

        for i in range(1,self.numClasses+1+5):
            heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
            im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('samples/WASPpose/heat/WASPpose_'+str(i)+'.png', im_heat)





    def test(self,epoch):
        self.model.eval()
        print("Testing") 
        
        img_path = '/home/bm3768/Desktop/Pose/Posezilla/samples/test_example3.jpg'
        center   = [184, 184]

        img  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
        img  = img.transpose(2, 0, 1)
        img  = torch.from_numpy(img)
        mean = [128.0, 128.0, 128.0]
        std  = [256.0, 256.0, 256.0]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)

        img = torch.unsqueeze(img, 0)

        self.model.eval()

        input_var = img.cuda()

        # heat, limbs, yolo = self.model(input_var, img_path)
        heat = self.model(input_var, img_path)

        im = cv2.imread(img_path)

        heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        kpts = uniPose_kpts(heat, self.dataset, img_h=368.0, img_w=368.0)
        # draw_paint(img_path, kpts, 0, epoch, self.model_arch, self.dataset)

        # heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        heat = heat.detach().cpu().numpy()

        heat = heat[0].transpose(1,2,0)


        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                for k in range(heat.shape[2]):
                    if heat[i,j,k] < 0:
                        heat[i,j,k] = 0
                    

        im       = cv2.resize(cv2.imread(img_path),(368,368))

        heatmap = []
        for i in range(self.numClasses+1+5):
            heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
            im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('samples/WASPpose/heat/WASPpose_'+str(i)+'.png', im_heat)
        

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
parser.add_argument('--dataset', type=str, dest='dataset', default='LSP')
parser.add_argument('--train_dir', default='/home/bm3768/Desktop/Pose/dataset/LSP/train/',type=str, dest='train_dir')
parser.add_argument('--val_dir', type=str, dest='val_dir', default='/home/bm3768/Desktop/Pose/dataset/LSP/val/')
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--model_arch', default='CPM', type=str)

starter_epoch =    0
epochs        =  100

args = parser.parse_args()

args.model_arch = 'WASPpose'

args.dataset    = 'NTID_small'

args.model_name = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_" + args.dataset

if args.dataset == 'LSP':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/LSP/train/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/LSP/val/'
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_LSP_best.pth.tar"
elif args.dataset == 'MPII':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/MPII/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/MPII/'
elif args.dataset == 'COCO':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/COCO/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/COCO/'
elif args.dataset == 'NTID':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/NTID/train/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/NTID/val/'
elif args.dataset == 'NTID_small':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/NTID_small/train/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/NTID_small/val/'
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_NTID_small_best.pth.tar"
elif args.dataset == 'PoseTrack':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/PoseTrack/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/PoseTrack/'
elif args.dataset == 'BBC':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/BBC/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/BBC/'
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_BBC_best.pth.tar"


trainer = Trainer(args)
for epoch in range(starter_epoch, epochs):
    trainer.training(epoch)
    trainer.validation(epoch)
#     trainer.test(epoch)
trainer.validation(0)
# trainer.test(0)
