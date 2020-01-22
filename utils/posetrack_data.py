# -*-coding:UTF-8-*-
from __future__ import print_function, absolute_import

import utils.Mytransforms as Mytransforms
import numpy as np
import random
import math
import json
import glob
import cv2
import os

import torch
import torch.utils.data as data


from utils.extra_utils.osutils import *
from utils.extra_utils.imutils import *
from utils.extra_utils.transforms import *


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def getBoundingBox(img, kpt, height, width, stride):
    

    # print(kpt.shape)

    for person in range(kpt.shape[0]): 
        x = np.zeros((kpt.shape[1],1))
        y = np.zeros((kpt.shape[1],1))
        for index in range(kpt.shape[1]):
            if float(kpt[person,index,1]) >= 0 or float(kpt[person,index,0]) >= 0:
                # print(person,index, kpt.shape, kpt[person,index], type(person), type(index))
                x[index] = [kpt[person,index,1]]
                y[index] = [kpt[person,index,0]]


        x_min = int(max(min(x), 0))
        x_max = int(min(max(x), width))
        y_min = int(max(min(y), 0))
        y_max = int(min(max(y), height))

        center_x = (x_min + x_max)/2
        center_y = (y_min + y_max)/2
        w        =  x_max - x_min
        h        =  y_max - y_min

        coord = []
        coord.append([min(int(center_y/stride),height/stride-1), min(int(center_x/stride),width/stride-1)])
        coord.append([min(int(y_min/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
        coord.append([min(int(y_min/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])
        coord.append([min(int(y_max/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
        coord.append([min(int(y_max/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])

        boxes = np.zeros((kpt.shape[0],int(height/stride), int(width/stride), 5), dtype=np.float32)
        for i in range(5):
            # resize from 368 to 46
            x = int(coord[i][0]) * 1.0
            y = int(coord[i][1]) * 1.0
            heat_map = guassian_kernel(size_h=height/stride, size_w=width/stride, center_x=x, center_y=y, sigma=3)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            boxes[person,:, :, i] = heat_map

    box = np.sum(boxes,axis=0)

    return box


class PoseTrack_Data(data.Dataset):
    def __init__(self, is_train, root_dir, sigma, stride, transformer=None):
        self.inp_res      = 368
        self.out_res      = 46
        self.scale_factor = 0.25
        self.rot_factor   = 30
        self.label_type   = 'Gaussian'
        self.stride       = stride
        self.transformer  = transformer
        self.sigma        = sigma
        self.root_dir     = root_dir
        self.is_train     = is_train
        self.transform    = transform

        self.train_list, self.val_list = [], []

        self.train_dir  = self.root_dir + 'images/train/'
        self.val_dir    = self.root_dir + 'images/val/'
        self.test_dir   = self.root_dir + 'images/test/'

        self.anno_train = self.root_dir + 'annotations/train/'
        self.anno_val   = self.root_dir + 'annotations/val/'
        self.anno_test  = self.root_dir + 'annotations/test/'


        self.labelFiles    = {}
        self.img_List      = {}
        self.box_head      = {}
        self.keypoints     = {}

        if is_train:
            self.labelFiles = [f for f in os.listdir(self.anno_train)]
            self.labelFiles.sort()

            count = 0
            masterCount = 0
            for i in range(len(self.labelFiles)):
                with open(self.anno_train + self.labelFiles[i]) as anno_file:
                    self.anno = json.load(anno_file)

                frame_id  = []
                file_name = []
                for j in range(len(self.anno['images'])):
                    if self.anno['images'][j]['is_labeled'] == True:
                        frame_id.append(self.anno['images'][j]['frame_id'])
                        file_name.append(self.anno['images'][j]['file_name'])

                count     = 0
                tempBbox  = []
                tempkpts  = []
                for j in range(len(self.anno['annotations'])):
                    if self.anno['annotations'][j]['image_id'] == frame_id[count]:
                        tempBbox.append(self.anno['annotations'][j]['bbox_head'])
                        tempkpts.append(self.anno['annotations'][j]['keypoints'])

                    elif self.anno['annotations'][j]['image_id'] != frame_id[count]\
                      and self.anno['annotations'][j-1]['image_id'] == frame_id[count]:
                        self.box_head[masterCount]  = tempBbox
                        self.keypoints[masterCount] = tempkpts
                        self.img_List[masterCount]  = file_name[count-1]

                        count  += 1
                        masterCount += 1
                        tempBbox  = []
                        tempkpts  = []

                        tempBbox.append(self.anno['annotations'][j]['bbox_head'])
                        tempkpts.append(self.anno['annotations'][j]['keypoints'])

                    else:
                        print(self.anno['annotations'][j]['image_id'])
        
                self.box_head[masterCount]  = tempBbox
                self.keypoints[masterCount] = tempkpts
                self.img_List[masterCount]  = file_name[count]

            self.train_list = self.img_List

            print("Images for train: ",len(self.train_list))

        else:
            self.labelFiles = [f for f in os.listdir(self.anno_val)]
            self.labelFiles.sort()

            count = 0
            masterCount = 0
            for i in range(len(self.labelFiles)):
                with open(self.anno_val + self.labelFiles[i]) as anno_file:
                    self.anno = json.load(anno_file)

                frame_id  = []
                file_name = []
                for j in range(len(self.anno['images'])):
                    if self.anno['images'][j]['is_labeled'] == True:
                        frame_id.append(self.anno['images'][j]['frame_id'])
                        file_name.append(self.anno['images'][j]['file_name'])

                count     = 0
                tempBbox  = []
                tempkpts  = []
                for j in range(len(self.anno['annotations'])):
                    if self.anno['annotations'][j]['image_id'] == frame_id[count]:
                        tempBbox.append(self.anno['annotations'][j]['bbox_head'])
                        tempkpts.append(self.anno['annotations'][j]['keypoints'])

                    elif self.anno['annotations'][j]['image_id'] != frame_id[count]\
                      and self.anno['annotations'][j-1]['image_id'] == frame_id[count]:
                        self.box_head[masterCount]  = tempBbox
                        self.keypoints[masterCount] = tempkpts
                        self.img_List[masterCount]  = file_name[count-1]

                        count  += 1
                        masterCount += 1
                        tempBbox  = []
                        tempkpts  = []

                        tempBbox.append(self.anno['annotations'][j]['bbox_head'])
                        tempkpts.append(self.anno['annotations'][j]['keypoints'])

                    else:
                        print(self.anno['annotations'][j]['image_id'])
        
                self.box_head[masterCount]  = tempBbox
                self.keypoints[masterCount] = tempkpts
                self.img_List[masterCount]  = file_name[count]

            self.val_list = self.img_List

            print("Images for val:   ",len(self.val_list))
                


    def __getitem__(self, index):
        if self.is_train:
            items = "/home/bm3768/Desktop/Pose/dataset/PoseTrack/" + self.train_list[index]
        else:
            items = "/home/bm3768/Desktop/Pose/dataset/PoseTrack/" + self.val_list[index]

        im = cv2.imread(items)
        if im is None:
            print(items)

        img = np.array(im,dtype=np.float32)
        kps = np.asarray(self.keypoints[index])

        center = {}

        center[0] = [img.shape[0]/2,img.shape[1]/2]

        # print("kps ", kps.shape)

        kpt = np.zeros((kps.shape[0],17,3))
        for i in range(kps.shape[0]):
            points = np.reshape(kps[i], (17,3))
            kpt[i] = points

        kpts = np.zeros((kpt.shape[0]*17,3))

        for i in range(kpt.shape[0]):
            kpts[17*i:17*(i+1),:] = kpt[i,:,:] 

        # print("Image ", img.shape)
        # print("Kpt ", kpt.shape)
        # print("Kpts ", kpts.shape)
        # print("Center ", center)


        img, kpts, center = self.transformer(img, kpts, center)

        for i in range(kpt.shape[0]):
            kpt[i,:,:] = kpts[17*i:17*(i+1),:]

        # kpt = torch.Tensor(kpt)

        # print("Image ", img.shape)
        # print("Kpt ", kpt.shape)
        # print("Center ", center)

        height, width, _ = img.shape

        # kpt = np.zeros((17,3))
        # for i in range(kpts.shape[0]):
        #     kpt = kpt + kpts[i,:,:]

        # print(kpt[:,2])

        # np.clip(kpt[:,2],0,1,kpt[:,2])

        # print(kpt[:,2])

        box = getBoundingBox(img, kpt, height, width, self.stride)

        heatmaps = np.zeros((kpt.shape[0],int(height/self.stride), int(width/self.stride), int(kpt.shape[1]+1)), dtype=np.float32)
        for i in range(kpt.shape[0]):
            for j in range(kpt.shape[1]):
                # resize from 368 to 46
                x = int(kpt[i,j,0]) * 1.0 / self.stride
                y = int(kpt[i,j,1]) * 1.0 / self.stride
                heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmaps[i,:, :, j + 1] = heat_map

            heatmaps[i,:, :, 0] = 1.0 - np.max(heatmaps[i,:, :, 1:], axis=2)  # for background

        # print(heatmaps.shape)

        # heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(kpt.shape[1]+1)), dtype=np.float32)

        heatmap = np.sum(heatmaps,axis=0)
        # print(heatmap.shape)

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0][0], center_y=center[0][1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        box       = Mytransforms.to_tensor(box)

        return img, heatmap, centermap, items, 0, box


    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.val_list)


        # #center = torch.Tensor(items['objpos'])
        # center = items['objpos']
        # scale  = items['scale_provided']

        # if center[0] != -1:
        #     center[1] = center[1] + 15*scale
        #     scale     = scale*1.25


        # nParts = pts.size(0)

        # img    = np.array(cv2.imread(img_path), dtype=np.float32)

    #     # expand dataset
    #     img, kpt, center = self.transformer(img, pts, center, scale)
    #     height, width, _ = img.shape

    #     heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(len(kpt)+1)), dtype=np.float32)
    #     for i in range(len(kpt)):
    #         # resize from 368 to 46
    #         x = int(kpt[i][0]) * 1.0 / self.stride
    #         y = int(kpt[i][1]) * 1.0 / self.stride
    #         heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
    #         heat_map[heat_map > 1] = 1
    #         heat_map[heat_map < 0.0099] = 0
    #         heatmap[:, :, i + 1] = heat_map

    #     heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

    #     centermap = np.zeros((height, width, 1), dtype=np.float32)
    #     center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
    #     center_map[center_map > 1] = 1
    #     center_map[center_map < 0.0099] = 0
    #     centermap[:, :, 0] = center_map

    #     img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
    #                                  [256.0, 256.0, 256.0])
    #     heatmap   = Mytransforms.to_tensor(heatmap)
    #     centermap = Mytransforms.to_tensor(centermap)

    #     return img, heatmap, centermap, img_path


    # def __len__(self):
    #     if self.is_train:
    #         return len(self.train_list)
    #     else:
    #         return len(self.val_list)