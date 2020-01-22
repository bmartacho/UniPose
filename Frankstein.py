# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
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
from model.CPM     import CPM

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
        self.test_dir     = args.test_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 1
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 1
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 1
        self.stride       = 8

        cudnn.benchmark   = True

        if self.dataset   ==  "LSP":
            self.numClasses  = 14
        elif self.dataset == "MPII":
            self.numClasses  = 16
        elif self.dataset == "COCO":
            self.numClasses  = 17
        elif self.dataset == "NTID":
            self.numClasses  = 19
        elif self.dataset == "PoseTrack":
            self.numClasses  = 17
        elif self.dataset == "BBC":
            self.numClasses  = 7


        self.train_loader, self.val_loader, self.test_loader = getDataloader(self.dataset, self.train_dir,\
            self.val_dir, self.test_dir, self.sigma, self.stride, self.workers, self.batch_size)

        if self.model_arch == "WASPpose":
            model = waspnet(self.dataset, num_classes=self.numClasses,backbone='resnet',\
                            output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)
        elif self.model_arch == "CPM":
            model = CPM(k=self.numClasses+5)

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
            prefix = 'invalid'

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


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)

        # for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):

        for i, (input, heatmap, centermap, img_path, original_image, segmented, box) in enumerate(tbar):

            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =      input.cuda()
            centermap_var =  centermap.cuda()
            heatmap_var   =    heatmap.cuda()
            # limbs_var     =   limbsmap.cuda()
            # box_var       =        box.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var, centermap_var)


            # target = torch.cat((heatmap_var, box_var), dim=1)

            heat = F.interpolate(heat, size=heatmap_var.size()[2:], mode='bilinear', align_corners=True)
            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat
            train_loss += loss_heat.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1

            # if i == 15:
            #     break


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
        # for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):

        for i, (input, heatmap, centermap, img_path, original_image, segmented, box) in enumerate(tbar):
            cnt += 1

            input_var     =     input.cuda()
            heatmap_var   =   heatmap.cuda()
            centermap_var = centermap.cuda()
            # limbs_var     =  limbsmap.cuda()
            # box_var       =       box.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var, centermap_var)

            # target = torch.cat((heatmap_var, box_var), dim=1)

            # heat = F.interpolate(heat, size=heatmap.size()[2:], mode='bilinear', align_corners=True)



            kk = F.interpolate(heatmap, size=segmented.size()[2:], mode='bilinear', align_corners=True)

            im = np.asarray(cv2.resize(cv2.imread(img_path[0]),(368,368)))
            
            kpts = get_kpts(kk, img_h=368, img_w=368)

            draw_paint(im, kpts, i, epoch, self.model_arch, self.dataset)


            # kk = kk.detach().cpu().numpy()

            # kk = kk[0].transpose(1,2,0)


            # for i in range(kk.shape[0]):
            #     for j in range(kk.shape[1]):
            #         for k in range(kk.shape[2]):
            #             if kk[i,j,k] < 0:
            #                 kk[i,j,k] = 0
                    

            # im = cv2.imread(img_path[0])


            # heatmap = []

            # cv2.imwrite('samples/WASPpose/heat/WASPpose_original.png', im)

            # background = -255*(kk[:,:,0]-np.amax(kk[:,:,0]))
            # heatmap = cv2.applyColorMap(np.uint8(background), cv2.COLORMAP_JET)
            # im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            # cv2.imwrite('samples/WASPpose/heat/WASPpose_0.png', im_heat)

            # for i in range(1,self.numClasses+1):
            #     heatmap = cv2.applyColorMap(np.uint8(255*kk[:,:,i]), cv2.COLORMAP_JET)
            #     im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            #     cv2.imwrite('samples/WASPpose/heat/WASPpose_'+str(i)+'.png', im_heat)


            # quit()






            loss_heat   = self.criterion(heat,  heatmap_var)




            loss = loss_heat
            val_loss += loss_heat.item()

            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.2,0.5, self.dataset)

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

            # if i == 15:
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



    def test(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss    = 0.0
        yolo_losses  = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK   = np.zeros(self.numClasses+1)
        PCKh  = np.zeros(self.numClasses+1)
        count = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):
            cnt += 1

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()
            # limbs_var     =   limbsmap.cuda()
            # box_var       =        box.cuda()
            centermap_var =  centermap.cuda()

            orig_img = limbsmap

            self.optimizer.zero_grad()

            # bottomRight = box
            # upperLeft   = limbsmap

            heat = self.model(input_var, centermap_var)

            # target = torch.cat((heatmap_var, box_var), dim=1)

            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat
            test_loss += loss_heat.item()

            tbar.set_description('Test  loss: %.6f' % (test_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.2,0.5, self.dataset)

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

            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            im = np.asarray(orig_img[0])
            
            kpts = get_kpts(heat, img_h=368, img_w=368)
            # gt = get_kpts(heatmap_var, img_h=368, img_w=368)

            # print(kpts)
            # print(gt)

            draw_paint(im, kpts, i, epoch, self.model_arch, self.dataset)

            # if i == 0:
            #     break

        heatmap_var = F.interpolate(heatmap_var, size=input_var.size()[2:], mode='bilinear', align_corners=True)
        heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
            
        PCKhAvg = PCKh.sum()/(self.numClasses+1)
        PCKAvg  =  PCK.sum()/(self.numClasses+1)

        if mAP > self.isBest:
            self.isBest = mAP
        #     save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name)
        #     print("Model saved to "+self.args.model_name)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK


        print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.isBest*100, self.bestPCK*100,self.bestPCKh*100))


        im = cv2.imread(img_path[0])

        im = np.asarray(orig_img[0])

        if self.dataset == "BBC":
            im = im[-368:,-368:,:]
        

        if self.dataset == "BBC" or self.dataset == "MPII":
            kpts = get_kpts(heat, img_h=368, img_w=368)
        elif self.dataset == "NTID":
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
        im = np.asarray(orig_img[0])

        if self.dataset == "BBC":
            im = im[-368:,-368:,:]

        heatmap = []

        cv2.imwrite('samples/WASPpose/heat/WASPpose_original.png', im)

        background = -255*(heat[:,:,0]-np.amax(heat[:,:,0]))
        heatmap = cv2.applyColorMap(np.uint8(background), cv2.COLORMAP_JET)
        im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
        cv2.imwrite('samples/WASPpose/heat/WASPpose_0.png', im_heat)

        for i in range(1,self.numClasses+1):
            heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
            im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('samples/WASPpose/heat/WASPpose_'+str(i)+'.png', im_heat)




    def single(self,epoch):
        self.model.eval()
        print("Single Image") 
        
        folder = '/home/bm3768/Desktop/Pose/dataset/BBC/BBCpose/13/'
        for frame in range(1,1800):
            print(int(100*frame/1800),"%")
            img_path  = folder + str(frame) + '.jpg'
            # img_path  = '/home/bm3768/Desktop/Pose/Posezilla/samples/IMG_2281.jpeg'
            # img_path  = '/home/bm3768/Desktop/Pose/dataset/PennAction/frames/0001/000001.jpg'

            if self.dataset == "LSP" or self.dataset == "MPII":
                img  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
            else:    
                img  = np.array(cv2.imread(img_path), dtype=np.float32)


            if self.dataset == "BBC":
                img= img[-368:,-368:,:]

            img  = img.transpose(2, 0, 1)
            img  = torch.from_numpy(img)
            mean = [128.0, 128.0, 128.0]
            std  = [256.0, 256.0, 256.0]
            for t, m, s in zip(img, mean, std):
                t.sub_(m).div_(s)

            img = torch.unsqueeze(img, 0)
            input_var = img.cuda()


            # print(input_var.shape)

            heat = self.model(input_var, torch.zeros((1,46,46,3)))

            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)


            if self.dataset == "LSP" or self.dataset == "MPII":
                im  = cv2.resize(cv2.imread(img_path),(368,368))
            elif self.dataset == "NTID":
                im  = cv2.resize(cv2.imread(img_path),(800,800))
            else:    
                im  = cv2.imread(img_path)

            if self.dataset == "BBC":
                im = im[-368:,-368:,:]


            kpts = get_kpts(heat, img_h=im.shape[0], img_w=im.shape[1])

            draw_paint(im, kpts, frame, epoch, self.model_arch, self.dataset)

            # Save one sample
            heat = heat.detach().cpu().numpy()

            heat = heat[0].transpose(1,2,0)


            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    for k in range(heat.shape[2]):
                        if heat[i,j,k] < 0:
                            heat[i,j,k] = 0
                    

            if self.dataset == "LSP" or self.dataset == "MPII":
                im  = cv2.resize(cv2.imread(img_path),(368,368))
            elif self.dataset == "NTID":
                im  = cv2.resize(cv2.imread(img_path),(800,800))
            else:    
                im  = cv2.imread(img_path)


            if self.dataset == "BBC":
                im = im[-368:,-368:,:]


            heatmap = []

            cv2.imwrite('samples/WASPpose/heat/WASPpose_original_'+str(frame)+'.png', im)

            background = -255*(heat[:,:,0]-np.amax(heat[:,:,0]))
            heatmap = cv2.applyColorMap(np.uint8(background), cv2.COLORMAP_JET)
            im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('samples/WASPpose/heat/0/'+str(frame)+'.png', im_heat)

            for i in range(1,self.numClasses+1):
                if not os.path.exists("samples/WASPpose/heat/"+str(i)):
                    os.makedirs("samples/WASPpose/heat/"+str(i))

                heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
                im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
                cv2.imwrite('samples/WASPpose/heat/'+str(i)+'/'+str(frame)+'.png', im_heat)
        


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
parser.add_argument('--dataset', type=str, dest='dataset', default='LSP')
parser.add_argument('--train_dir', default=None,type=str, dest='train_dir')
parser.add_argument('--val_dir', type=str, dest='val_dir', default=None)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--model_arch', default='CPM', type=str)

starter_epoch =   55
epochs        =   100

args = parser.parse_args()

args.model_arch = 'WASPpose'

args.dataset    = 'MPII'

args.model_name = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPpose_with_BBOX_" + args.dataset

if args.dataset == 'LSP':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/LSP/train/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/LSP/val/'
    args.test_dir   = None
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_LSP_best.pth.tar"

elif args.dataset == 'NTID':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/NTID_small/train/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/NTID_small/val/'
    args.test_dir    = '/home/bm3768/Desktop/Pose/dataset/NTID_small/test/'
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPpose_with_BBOX_NTID_best.pth.tar"

elif args.dataset == 'MPII':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/MPII/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/MPII/'
    args.test_dir   = '/home/bm3768/Desktop/Pose/dataset/MPII/'
    args.pretrained = '/home/bm3768/Desktop/Pose/Posezilla/ckpt/Best_Results/WASPpose_with_BBOX_MPII_best_3.pth.tar'

elif args.dataset == 'PoseTrack':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/PoseTrack/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/PoseTrack/'

elif args.dataset == 'BBC':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/BBC/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/BBC/'
    args.test_dir   = '/home/bm3768/Desktop/Pose/dataset/BBC/'
    args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_with_BBOX_BBC_best.pth.tar"


trainer = Trainer(args)
# for epoch in range(starter_epoch, epochs):
#     trainer.training(epoch)
#     trainer.validation(epoch)
    # trainer.test(epoch)
    # trainer.single(epoch)
# trainer.validation(0)
trainer.test(0)
# trainer.single(0)
