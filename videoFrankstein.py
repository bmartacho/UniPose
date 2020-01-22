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
import warnings
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import save_checkpoint      as save_checkpoint
from utils.utils import printAccuracies      as printAccuracies
from utils.utils import guassian_kernel      as guassian_kernel
from utils.utils import get_parameters       as get_parameters
from utils       import Mytransforms         as  Mytransforms 
from utils.utils import getDataloader        as getDataloader
from utils.utils import getOutImages         as getOutImages
from utils.utils import AverageMeter         as AverageMeter
from utils.utils import draw_paint           as draw_paint
from utils       import evaluate             as evaluate
from utils.utils import get_kpts             as get_kpts

from model.VideoWaspnet import waspnet

from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

from PIL import Image



class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset
        self.frame_memory = args.frame_memory


        self.workers      = 1
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 1
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 1

        cudnn.benchmark   = True

        if self.dataset   ==  "Penn_Action":
            self.numClasses  = 13

        self.train_loader, self.val_loader, self.test_loader = getDataloader(self.dataset, self.train_dir,\
                                self.val_dir, None, self.sigma, self.stride, self.workers, self.frame_memory)

        model = waspnet(num_classes=self.numClasses,backbone='resnet',output_stride=16,\
                        sync_bn=True,freeze_bn=False, stride=self.stride)

        self.model       = model.cuda()

        self.criterion   = nn.MSELoss().cuda()

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.best_model  = 12345678.9

        self.iters       = 0

        if self.args.pretrained is not None:
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']

            prefix = 'decoder.last_conv.8'

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

        for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =       input.cuda()
            heatmap_var   =     heatmap.cuda()
            centermap_var =   centermap.cuda()
            box_var       =         box.cuda()

            self.optimizer.zero_grad()

            target = torch.cat((heatmap_var, box_var), dim=2)

            heat = torch.zeros(self.numClasses+1+5, input.shape[3], input.shape[4]).cuda()
            cell = torch.zeros(39, input.shape[3], input.shape[4]).cuda()
            hide = torch.zeros(39, input.shape[3], input.shape[4]).cuda()

            losses = {}
            loss   = 0

            start_model = time.time()
            for j in range(self.frame_memory):
                heat, cell, hide = self.model(input_var, centermap_var, j, heat, hide, cell)

                losses[j] = self.criterion(heat[0], target[0][j])
                loss     += losses[j]

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1


    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK   = np.zeros(self.numClasses+1)
        PCKh  = np.zeros(self.numClasses+1)
        count = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, centermap, img_path, limbsmap, box) in enumerate(tbar):

            cnt += 1

            input_var     =       input.cuda()
            heatmap_var   =     heatmap.cuda()
            centermap_var =   centermap.cuda()
            box_var       =         box.cuda()

            self.optimizer.zero_grad()

            target = torch.cat((heatmap_var, box_var), dim=2)

            heat = torch.zeros(self.numClasses+1+5, input.shape[3], input.shape[4]).cuda()
            cell = torch.zeros(39, input.shape[3], input.shape[4]).cuda()
            hide = torch.zeros(39, input.shape[3], input.shape[4]).cuda()

            losses = {}
            loss   = 0

            start_model = time.time()
            for j in range(self.frame_memory):
                heat, cell, hide = self.model(input_var, centermap_var, j, heat, hide, cell)

                losses[j] = self.criterion(heat, target)

                loss  += losses[j].item()

                # tgt    = torch.zeros(1, target.shape[2], target.shape[3], target.shape[4]).cuda()

                # tgt[0]    = target[0][j]

                # print(target[:,j].shape, tgt.shape)

                acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(),\
                                                       target[:,j].detach().cpu().numpy(),0.2,0.5, self.dataset)

                AP[0]     = (AP[0]  *(self.frame_memory*i+j) + acc[0])      / ((self.frame_memory*i+j) + 1)
                PCK[0]    = (PCK[0] *(self.frame_memory*i+j) + acc_PCK[0])  / ((self.frame_memory*i+j) + 1)
                PCKh[0]   = (PCKh[0]*(self.frame_memory*i+j) + acc_PCKh[0]) / ((self.frame_memory*i+j) + 1)

                for k in range(self.numClasses+1):
                    if visible[k] == 1:
                        AP[k]     = (AP[k]  *count[k] + acc[k])      / (count[k] + 1)
                        PCK[k]    = (PCK[k] *count[k] + acc_PCK[k])  / (count[k] + 1)
                        PCKh[k]   = (PCKh[k]*count[k] + acc_PCKh[k]) / (count[k] + 1)
                        count[k] += 1

            mAP     =   AP[1:].sum()/(self.numClasses)
            mPCK    =  PCK[1:].sum()/(self.numClasses)
            mPCKh   = PCKh[1:].sum()/(self.numClasses)


            val_loss += loss


            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))


        #printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
            
        PCKhAvg = PCKh.sum()/(self.numClasses+1)
        PCKAvg  =  PCK.sum()/(self.numClasses+1)

        if mAP > self.isBest:
            self.isBest = mAP
            save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK

        print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.isBest*100, self.bestPCK*100,self.bestPCKh*100))



    def test(self,epoch):
        self.model.eval()
        print("Testing") 
        
        img_path = '/home/bm3768/Desktop/Pose/Artacho/samples/test_example2.png'
        center   = [184, 184]

        img  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
        img  = img.transpose(2, 0, 1)
        img  = torch.from_numpy(img)
        mean = [128.0, 128.0, 128.0]
        std  = [256.0, 256.0, 256.0]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)

        img       = torch.unsqueeze(img, 0)

        self.model.eval()

        input_var     =       img.cuda()


        heat = self.model(input_var)

        heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
        draw_paint(img_path, kpts, 0, epoch, self.model_arch, self.dataset)

        heat = heat.detach().cpu().numpy()

        heat = heat[0].transpose(1,2,0)


        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                for k in range(heat.shape[2]):
                    if heat[i,j,k] < 0:
                        heat[i,j,k] = 0
                    

        im       = cv2.resize(cv2.imread(img_path),(368,368))

        heatmap = []
        for i in range(self.numClasses+1):
            heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
            im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('samples/WASPpose/heat/WASPpose_'+str(i)+'.png', im_heat)
        

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
parser.add_argument('--dataset', type=str, dest='dataset', default='LSP')
parser.add_argument('--train_dir', default=None,type=str, dest='train_dir')
parser.add_argument('--val_dir', type=str, dest='val_dir', default=None)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--model_arch', default='CPM', type=str)

starter_epoch =    0
epochs        =  100

args = parser.parse_args()

# args.pretrained = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/Best_Results/WASPnet_Video_4_Penn_Action_best.pth.tar"

args.model_arch = 'WASPpose'

args.dataset    = 'Penn_Action'

args.frame_memory = 3

args.model_name = "/home/bm3768/Desktop/Pose/Posezilla/ckpt/WASPnet_Video_"+str(args.frame_memory)+"_" + args.dataset

if args.dataset == 'Penn_Action':
    args.train_dir  = '/home/bm3768/Desktop/Pose/dataset/PennAction/'
    args.val_dir    = '/home/bm3768/Desktop/Pose/dataset/PennAction/'


trainer = Trainer(args)
for epoch in range(starter_epoch, epochs):
    trainer.training(epoch)
    trainer.validation(epoch)
#    trainer.test(epoch)
trainer.validation(0)
#trainer.test(0)
