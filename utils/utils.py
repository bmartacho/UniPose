from collections import namedtuple

from utils       import lsp_lspet_data       as lsp_lspet_data
from utils       import mpii_data            as mpii_data
from utils       import mpii_data            as Mpii
# from utils       import coco_data            as coco_data
from utils       import penn_action_data     as penn_action
from utils       import ntid_data            as ntid_data
from utils       import posetrack_data       as posetrack_data
from utils       import bbc_data             as bbc_data
import utils.Mytransforms as Mytransforms
import torch.nn.functional as F
import math
import torch
import shutil
import time
import os
import random
from easydict import EasyDict as edict
import yaml
import numpy as np
import cv2
import time

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, iters, base_lr, gamma, step_size, policy='step', multiple=[1]):

    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (gamma ** (iters // step_size))

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

def save_checkpoint(state, is_best, filename='checkpoint'):

    if is_best:
        torch.save(state, filename + '_best.pth.tar')

def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    #for x in parser:
    #    print('{}: {}'.format(x, parser[x]))
    return parser


def get_parameters(model, lr, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': lr},
            {'params': lr_2, 'lr': lr * 2.},
            {'params': lr_4, 'lr': lr * 4.},
            {'params': lr_8, 'lr': lr * 8.}]

    return params, [1., 2., 4., 8.]


def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts


def draw_paint(im, kpts, mapNumber, epoch, model_arch, dataset):

           #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000,000,255], [000,255,000], [000,000,255], [255,255,000], [255,255,000], [255,000,255], [000,255,000],\
              [255,000,000], [255,255,000], [255,000,255], [000,255,000], [000,255,000], [000,000,255], [255,255,000], [255,000,000]]
           #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE

    if dataset == "LSP":
        limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10,11], [7, 6], [12, 3],\
                    [12, 2], [ 2, 1], [ 1, 0], [ 3, 4], [4,  5], [15,16], [16,18], [17,18], [15,17]]
        kpts[15][0] = kpts[15][0]  - 25
        kpts[15][1] = kpts[15][1]  - 50
        kpts[16][0] = kpts[16][0]  - 25
        kpts[16][1] = kpts[16][1]  + 50
        kpts[17][0] = kpts[17][0] + 25
        kpts[17][1] = kpts[17][1] - 50
        kpts[18][0] = kpts[18][0] + 25
        kpts[18][1] = kpts[18][1] + 50


    elif dataset == "MPII":
                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM   TORSO    L.HIP   L.THIGH   L.CALF   R.HIP   R.THIGH   R.CALF  EXT.HEAD
        limbSeq = [[ 8, 9], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 6, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 7, 8]]

    elif dataset == "NTID":
        limbSeq = [[ 0, 1], [ 1, 2], [ 2, 3], [ 2, 4], [ 4, 5], [ 5, 6], [ 2, 8], [ 8, 9],\
                   [ 9,10], [ 0,12], [ 0,13],[20,21],[21,23],[20,22],[22,23]]
        kpts[20][0] = kpts[20][0]  - 25
        kpts[20][1] = kpts[20][1]  - 50
        kpts[21][0] = kpts[21][0]  - 25
        kpts[21][1] = kpts[21][1]  + 50
        kpts[22][0] = kpts[22][0] + 25
        kpts[22][1] = kpts[22][1] - 50
        kpts[23][0] = kpts[23][0] + 25
        kpts[23][1] = kpts[23][1] + 50
    

                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM   TORSO    L.HIP   L.THIGH   L.CALF   R.HIP   R.THIGH   R.CALF  EXT.HEAD
        limbSeq = [[ 8, 7], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 5, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 8, 7]]

    elif dataset == "BBC":
                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM
        limbSeq = [[ 0,12], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6], [ 5, 6], [8,9],[8,10],[10,11],[9,11]]
        kpts.append([int((kpts[5][0]+kpts[6][0])/2),int((kpts[5][1]+kpts[6][1])/2)])
        kpts[8][0]  = kpts[8][0]  - 25
        kpts[8][1]  = kpts[8][1]  - 50
        kpts[9][0]  = kpts[9][0]  - 25
        kpts[9][1]  = kpts[9][1]  + 50
        kpts[10][0] = kpts[10][0] + 25
        kpts[10][1] = kpts[10][1] - 50
        kpts[11][0] = kpts[11][0] + 25
        kpts[11][1] = kpts[11][1] + 50

        
        colors = [[000,255,000], [000,000,255], [255,000,000], [000,255,000], [255,255,51], [255,000,255],\
                  [000,000,255], [000,000,255], [000,000,255], [000,000,255]]


    # im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        # mX = np.mean([X0, X1])
        # mY = np.mean([Y0, Y1])
        # length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        # angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        # cv2.fillConvexPoly(cur_im, polygon, colors[i])
        # if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
        #     im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

        if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
            if i<len(limbSeq)-4:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), colors[i], 5)
            else:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), [0,0,255], 5)

        im = cv2.addWeighted(im, 0.2, cur_im, 0.8, 0)

    cv2.imwrite('samples/WASPpose/Pose/'+str(mapNumber)+'.png', im)


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width      = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx               = np.argmax(heatmaps_reshaped, 2)
    maxvals           = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx     = idx.reshape((batch_size, num_joints, 1))

    preds   = np.tile(idx, (1,1,2)).astype(np.float32)

    preds[:,:,0] = (preds[:,:,0]) % width
    preds[:,:,1] = np.floor((preds[:,:,1]) / width)

    pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
    pred_mask    = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def getDataloader(dataset, train_dir, val_dir, test_dir, sigma, stride, workers, batch_size):
    if dataset == 'LSP':
        train_loader = torch.utils.data.DataLoader(
                                            lsp_lspet_data.LSP_Data('lspet', train_dir, sigma, stride,
                                            Mytransforms.Compose([Mytransforms.RandomHorizontalFlip(),])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)   
    
        val_loader   = torch.utils.data.DataLoader(
                                            lsp_lspet_data.LSP_Data('lsp', val_dir, sigma, stride,
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)

        test_loader  = 0

    elif dataset == 'MPII':
        train_loader = torch.utils.data.DataLoader(
                                            mpii_data.mpii(train_dir, sigma, "Train",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            mpii_data.mpii (val_dir, sigma, "Val",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)
    
        test_loader   = torch.utils.data.DataLoader(
                                            mpii_data.mpii (test_dir, sigma, "Val",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)

#     elif dataset == 'COCO':
#         train_loader = torch.utils.data.DataLoader(
#                                             coco_data.COCO_Data(True, train_dir, sigma, stride,
#                                             Mytransforms.Compose([Mytransforms.RandomResized(),
#                                             Mytransforms.RandomRotate(40),
#                                             #Mytransforms.RandomCrop(368),
#                                             Mytransforms.SinglePersonCrop(368),
#                                             Mytransforms.RandomHorizontalFlip(),])),
#                                             batch_size  = batch_size, shuffle=True,
#                                             num_workers = workers, pin_memory=True)
    
#         val_loader   = torch.utils.data.DataLoader(
#                                             coco_data.COCO_Data(False, val_dir, sigma, stride,
#                                             Mytransforms.Compose([Mytransforms.TestResized(368),
#                                             Mytransforms.SinglePersonCrop(368),])),
#                                             batch_size  = 1, shuffle=True,
#                                             num_workers = workers, pin_memory=True)

    elif dataset == 'Penn_Action':
        train_loader = torch.utils.data.DataLoader(
                                            penn_action.Penn_Action(train_dir, sigma, batch_size, True,
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            penn_action.Penn_Action(val_dir, sigma, batch_size, False,
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)

        test_loader = None

    elif dataset == 'NTID':
        train_loader = torch.utils.data.DataLoader(
                                            ntid_data.NTID(train_dir, sigma, "Train",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),
                                            Mytransforms.RandomHorizontalFlip_NTID(),])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            ntid_data.NTID (val_dir, sigma, "Val",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)
    
        test_loader  = torch.utils.data.DataLoader(
                                            ntid_data.NTID (test_dir, sigma, "Test",),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)

    elif dataset == 'PoseTrack':
        train_loader = torch.utils.data.DataLoader(
                                            posetrack_data.PoseTrack_Data(True, train_dir, sigma, stride,
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            posetrack_data.PoseTrack_Data(False, val_dir, sigma, stride,
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)

    elif dataset == "BBC":
        train_loader = torch.utils.data.DataLoader(
                                            bbc_data.BBC(train_dir, sigma, "Train",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),
                                            Mytransforms.RandomHorizontalFlip_NTID(),])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            bbc_data.BBC (val_dir, sigma, "Val",
                                            Mytransforms.Compose([Mytransforms.TestResized(368),])),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)
    
        test_loader   = torch.utils.data.DataLoader(
                                            bbc_data.BBC (val_dir, sigma, "Test",),
                                            batch_size  = 1, shuffle=True,
                                            num_workers = 1, pin_memory=True)


    return train_loader, val_loader, test_loader


def printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, dataset):
    if dataset == "LSP":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:     Void = %2.2f%%, Right Ankle = %2.2f%%,  Right Knee = %2.2f%%,   Right Hip = %2.2f%%,       Left Hip = %2.2f%%,"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100))
        print("    Left Knee = %2.2f%%,  Left Ankle = %2.2f%%, Right Wrist = %2.2f%%, Right Elbow = %2.2f%%, Right Shoulder = %2.2f%%,"\
            % (AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100))
        print("Left Shoulder = %2.2f%%,  Left Elbow = %2.2f%%,  Left Wrist = %2.2f%%,        Neck = %2.2f%%,       Head Top = %2.2f%%"\
            % (AP[10]*100,AP[11]*100,AP[12]*100,AP[13]*100,AP[14]*100))


        print("mPCK:  %.2f%%" % (mPCK*100))
        print("PCKs:    Void = %2.2f%%, Right Ankle = %2.2f%%,  Right Knee = %2.2f%%,   Right Hip = %2.2f%%,       Left Hip = %2.2f%%,"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100))
        print("    Left Knee = %2.2f%%,  Left Ankle = %2.2f%%, Right Wrist = %2.2f%%, Right Elbow = %2.2f%%, Right Shoulder = %2.2f%%,"\
            % (PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100))
        print("Left Shoulder = %2.2f%%,  Left Elbow = %2.2f%%,  Left Wrist = %2.2f%%,        Neck = %2.2f%%,       Head Top = %2.2f%%"\
            % (PCK[10]*100,PCK[11]*100,PCK[12]*100,PCK[13]*100,PCK[14]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("PCKhs:   Void = %2.2f%%, Right Ankle = %2.2f%%,  Right Knee = %2.2f%%,   Right Hip = %2.2f%%,       Left Hip = %2.2f%%,"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100))
        print("    Left Knee = %2.2f%%,  Left Ankle = %2.2f%%, Right Wrist = %2.2f%%, Right Elbow = %2.2f%%, Right Shoulder = %2.2f%%,"\
            % (PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100))
        print("Left Shoulder = %2.2f%%,  Left Elbow = %2.2f%%,  Left Wrist = %2.2f%%,        Neck = %2.2f%%,       Head Top = %2.2f%%"\
            % (PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100,PCKh[14]*100))

    elif dataset == "MPII":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:   %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100,PCKh[10]*100,\
                AP[11]*100,AP[12]*100,AP[13]*100,AP[14]*100,AP[15]*100,AP[16]*100))

        print("mPCK:  %.2f%%" % (mPCK*100))
        print("PCKs:  %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100,PCK[10]*100,\
                PCK[11]*100,PCK[12]*100,PCK[13]*100,PCK[14]*100,PCK[15]*100,PCK[16]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("PCKhs: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100,\
                PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100,PCKh[14]*100,PCKh[15]*100,PCKh[16]*100))

    elif dataset == "COCO":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:   %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100,PCKh[10]*100,\
                 AP[11]*100,AP[12]*100,AP[13]*100,AP[14]*100,AP[15]*100,AP[16]*100,AP[17]*100))

        print("mPCK:  %.2f%%" % (mPCK*100))
        print("PCKs:  %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100,PCK[10]*100,\
                PCK[11]*100,PCK[12]*100,PCK[13]*100,PCK[14]*100,PCK[15]*100,PCK[16]*100,PCKh[17]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("PCKhs: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100,\
                PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100,PCKh[14]*100,PCKh[15]*100,PCKh[16]*100,PCKh[17]*100))

    elif dataset == "Penn_Action":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:   %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100,PCKh[10]*100,\
                AP[11]*100,AP[12]*100,AP[13]*100))

        print("mPCK:  %.2f%%" % (mPCK*100))
        print("PCKs:  %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100,PCK[10]*100,\
                PCK[11]*100,PCK[12]*100,PCK[13]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("PCKhs: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100,\
                PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100))

    elif dataset == "NTID":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:     Void = %2.2f%%,     Spine Base = %2.2f%%,   Spine Mid = %2.2f%%,           Neck = %2.2f%%,        Head = %2.2f%%,"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100))
        print("Shoulder Left = %2.2f%%,     Elbow Left = %2.2f%%,  Wrist Left = %2.2f%%,  Hand Tip Left = %2.2f%%,    Hip Left = %2.2f%%,"\
            % (AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[13]*100))
        print("Shoulder Right = %2.2f%%,   Elbow Right = %2.2f%%, Wrist Right = %2.2f%%, Hand Tip Right = %2.2f%%,   Hip Right = %2.2f%%"\
            % (AP[9]*100,AP[10]*100,AP[11]*100,AP[12]*100,AP[14]*100))
        print("Spine Shoulder = %2.2f%%, Hand Tip Left = %2.2f%%,  Thumb Left = %2.2f%%, Hand Tip Right = %2.2f%%, Thumb Right = %2.2f%%"\
            % (AP[15]*100,AP[16]*100,AP[17]*100,AP[18]*100,AP[19]*100))


        print("mPCK:  %.2f%%" % (mPCK*100))
        print("APs:     Void = %2.2f%%,     Spine Base = %2.2f%%,   Spine Mid = %2.2f%%,           Neck = %2.2f%%,        Head = %2.2f%%,"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100))
        print("Shoulder Left = %2.2f%%,     Elbow Left = %2.2f%%,  Wrist Left = %2.2f%%,  Hand Tip Left = %2.2f%%,    Hip Left = %2.2f%%,"\
            % (PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[13]*100))
        print("Shoulder Right = %2.2f%%,   Elbow Right = %2.2f%%, Wrist Right = %2.2f%%, Hand Tip Right = %2.2f%%,   Hip Right = %2.2f%%"\
            % (PCK[9]*100,PCK[10]*100,PCK[11]*100,PCK[12]*100,PCK[14]*100))
        print("Spine Shoulder = %2.2f%%, Hand Tip Left = %2.2f%%,  Thumb Left = %2.2f%%, Hand Tip Right = %2.2f%%, Thumb Right = %2.2f%%"\
            % (PCK[15]*100,PCK[16]*100,PCK[17]*100,PCK[18]*100,PCK[19]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("APs:     Void = %2.2f%%,     Spine Base = %2.2f%%,   Spine Mid = %2.2f%%,           Neck = %2.2f%%,        Head = %2.2f%%,"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100))
        print("Shoulder Left = %2.2f%%,     Elbow Left = %2.2f%%,  Wrist Left = %2.2f%%,  Hand Tip Left = %2.2f%%,    Hip Left = %2.2f%%,"\
            % (PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[13]*100))
        print("Shoulder Right = %2.2f%%,   Elbow Right = %2.2f%%, Wrist Right = %2.2f%%, Hand Tip Right = %2.2f%%,   Hip Right = %2.2f%%"\
            % (PCKh[9]*100,PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[14]*100))
        print("Spine Shoulder = %2.2f%%, Hand Tip Left = %2.2f%%,  Thumb Left = %2.2f%%, Hand Tip Right = %2.2f%%, Thumb Right = %2.2f%%"\
            % (PCKh[15]*100,PCKh[16]*100,PCKh[17]*100,PCKh[18]*100,PCKh[19]*100))

    elif dataset == "BBC":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs: BG = %2.2f%%, HD = %2.2f%%, LH = %2.2f%%, RH = %2.2f%%, LE = %2.2f%% RE = %2.2f%%, LS = %2.2f%%, RS = %2.2f%%,"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100))


        print("mPCK:  %.2f%%" % (mPCK*100))
        print("APs: BG = %2.2f%%, HD = %2.2f%%, LH = %2.2f%%, RH = %2.2f%%, LE = %2.2f%% RE = %2.2f%%, LS = %2.2f%%, RS = %2.2f%%,"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100))
        
        print("mPCKh:  %.2f%%" % (mPCKh*100))
        print("APs: BG = %2.2f%%, HD = %2.2f%%, LH = %2.2f%%, RH = %2.2f%%, LE = %2.2f%% RE = %2.2f%%, LS = %2.2f%%, RS = %2.2f%%,"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100))



def getOutImages(heat, input_var, img_path, outName):
    heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

    heat = heat.detach().cpu().numpy()

    heat = heat[0].transpose(1,2,0)


    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            for k in range(heat.shape[2]):
                if heat[i,j,k] < 0:
                    heat[i,j,k] = 0
                

    im       = cv2.resize(cv2.imread(img_path[0]),(368,368))

    heatmap = []
    for i in range(15):
        heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
        im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
        cv2.imwrite('samples/WASPpose/heat/'+outName+'_'+str(i)+'.png', im_heat)



def draw_BBox(box, img_path, heatmap, input):
    image = input.detach().cpu().numpy()

    center_x   = box[0][0].detach().cpu().numpy()
    center_y   = box[0][1].detach().cpu().numpy()
    box_width  = box[0][2].detach().cpu().numpy()
    box_height = box[0][3].detach().cpu().numpy()

    # print(center_x, center_y, box_width, box_height)
    # print(img_path)

    img  = cv2.resize(cv2.imread(img_path[0]),(368,368))

    # draw points
    cv2.circle(img, (center_x, center_y), radius=1, thickness=-1, color=(0, 0, 255))

    [Y0, X0] = [center_y-box_height/2, center_x-box_width/2]
    [Y1, X1] = [center_y-box_height/2, center_x+box_width/2]
    # print([Y0, X0], [Y1, X1])
    mX = np.mean([X0, X1])
    mY = np.mean([Y0, Y1])
    length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(img, polygon, (0, 0, 255))

    [Y0, X0] = [center_y-box_height/2, center_x-box_width/2]
    [Y1, X1] = [center_y+box_height/2, center_x-box_width/2]
    # print([Y0, X0], [Y1, X1])
    mX = np.mean([X0, X1])
    mY = np.mean([Y0, Y1])
    length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(img, polygon, (0, 0, 255))

    [Y0, X0] = [center_y+box_height/2, center_x-box_width/2]
    [Y1, X1] = [center_y+box_height/2, center_x+box_width/2]
    # print([Y0, X0], [Y1, X1])
    mX = np.mean([X0, X1])
    mY = np.mean([Y0, Y1])
    length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(img, polygon, (0, 0, 255))

    [Y0, X0] = [center_y-box_height/2, center_x+box_width/2]
    [Y1, X1] = [center_y+box_height/2, center_x+box_width/2]
    # print([Y0, X0], [Y1, X1])
    mX = np.mean([X0, X1])
    mY = np.mean([Y0, Y1])
    length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(img, polygon, (0, 0, 255))

    kpts = get_kpts(heatmap, img_h=368.0, img_w=368.0)
    limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10, 11], [7, 6], [12, 3], [12, 2], [ 2, 1], [ 1, 0], [ 3, 4], [4,  5]]

    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(img, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, (0, 0, 255))


    cv2.imwrite('samples/bbox.png', img)


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    print(prediction.shape)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :,  0] = prediction[:, :,  0] - prediction[:, :, 2] / 2
    box_corner[:, :,  1] = prediction[:, :,  1] - prediction[:, :, 3] / 2
    box_corner[:, :,  2] = prediction[:, :,  0] + prediction[:, :, 2] / 2
    box_corner[:, :,  3] = prediction[:, :,  1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    print(prediction.shape)

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua



def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details
