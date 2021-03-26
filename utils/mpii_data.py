# -*-coding:UTF-8-*-
from __future__ import print_function, absolute_import

import utils.Mytransforms as Mytransforms
import numpy as np
import random
import scipy
import math
import json
import glob
import cv2
import sys
import os

import torch
import torch.utils.data as data


def get_transform(center, scale, resolution):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(resolution[1]) / h
    t[1, 1] = float(resolution[0]) / h
    t[0, 2] = resolution[1] * (-float(center[0]) / h + .5)
    t[1, 2] = resolution[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    return t


def transformImage(pt, center, scale, resolution):
    t = get_transform(center, scale, resolution)
    t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)

    return new_pt[:2].astype(int) + 1


def crop(img, points, center, scale, resolution):
    upperLeft   = np.array(transformImage([0, 0], center, scale, resolution))
    bottomRight = np.array(transformImage(resolution, center, scale, resolution))

    # Range to fill new array
    new_x = max(0, -upperLeft[0]), min(bottomRight[0], img.shape[1]) - upperLeft[0]
    new_y = max(0, -upperLeft[1]), min(bottomRight[1], img.shape[0]) - upperLeft[1]
    # Range to sample from original image
    old_x = max(0, upperLeft[0]), min(img.shape[1], bottomRight[0])
    old_y = max(0, upperLeft[1]), min(img.shape[0], bottomRight[1])
    new_img = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]


    points[:,0] = points[:,0] - max(0, upperLeft[0])# + max(0, -upperLeft[0])
    points[:,1] = points[:,1] - max(0, upperLeft[1])# + max(0, -upperLeft[1])

    center[0] -= max(0, upperLeft[0])
    center[1] -= max(0, upperLeft[1])

    return new_img, upperLeft, bottomRight, points, center


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class mpii(data.Dataset):
    def __init__(self, root_dir, sigma, is_train, transform=None):
        self.width       = 368
        self.height      = 368
        self.transformer = transform
        self.is_train    = is_train
        self.sigma       = sigma
        self.parts_num   = 16
        self.stride      = 8

        self.labels_dir  = root_dir
        self.images_dir  = root_dir + 'images/'

        self.videosFolders = {}
        self.labelFiles    = {}
        self.full_img_List = {}
        self.numPeople     = []


        with open(self.labels_dir+"mpii_annotations.json") as anno_file:
            self.anno = json.load(anno_file)

        self.train_list = []
        self.val_list   = []

        for idx,val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.val_list.append(idx)
            else:
                self.train_list.append(idx)


        if is_train == "Train":
            self.img_List = self.train_list
            print("Train images ",len(self.img_List))

        elif is_train == "Val":
            self.img_List = self.val_list
            print("Val   images ",len(self.img_List))


    def __getitem__(self, index):
        scale_factor = 0.25

        variable = self.anno[self.img_List[index]]
        
        while not os.path.isfile(self.labels_dir + variable['img_paths'][:-4]+'.png'):
            index = index - 1
            variable = self.anno[self.img_List[index]]

        img_path  = self.images_dir + variable['img_paths']
        
        # BBox was added to the labels by the authors to perform additional training and testing, as referred in the paper.
        # Intentionally left as comment since it is not part of the dataset.
#         bbox      = np.load(self.labels_dir + "BBOX/" + variable['img_paths'][:-4] + '.npy')

        points   = torch.Tensor(variable['joint_self'])
        center   = torch.Tensor(variable['objpos'])
        scale    = variable['scale_provided']


        if center[0] != -1:
            center[1] = center[1] + 15*scale
            scale     = scale*1.25


        # Single Person
        nParts = points.size(0)
        img    = cv2.imread(img_path)
#         box    = np.zeros((2,2))

#         for i in range(bbox.shape[0]):
#             if center[0] > bbox[i,0] and center[0] < bbox[i,2] and\
#                center[1] > bbox[i,1] and center[1] < bbox[i,3]:

#                upperLeft   = bbox[i,0:2].astype(int)
#                bottomRight = bbox[i,-2:].astype(int)
#                box = bbox[i,:]

#                img[:,0:upperLeft[0],:]  = np.ones(img[:,0:upperLeft[0],:].shape) *255
#                img[0:upperLeft[1],:,:]  = np.ones(img[0:upperLeft[1],:,:].shape) *255
#                img[:,bottomRight[0]:,:] = np.ones(img[:,bottomRight[0]:,:].shape)*255
#                img[bottomRight[1]:,:,:] = np.ones(img[bottomRight[1]:,:,:].shape)*255

#                break

        # img, upperLeft, bottomRight, points, center = crop(img, points, center, scale, [self.height, self.width])

        kpt = points

        # img, kpt, center = self.transformer(img, points, center)
        if img.shape[0] != 368 or img.shape[1] != 368:
            kpt[:,0] = kpt[:,0] * (368/img.shape[1])
            kpt[:,1] = kpt[:,1] * (368/img.shape[0])
            img = cv2.resize(img,(368,368))
        height, width, _ = img.shape

        heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(len(kpt)+1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height/self.stride),size_w=int(width/self.stride), center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        centermap = np.zeros((int(height/self.stride), int(width/self.stride), 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=int(height/self.stride), size_w=int(width/self.stride), center_x=int(center[0]/self.stride), center_y=int(center[1]/self.stride), sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        orig_img = cv2.imread(img_path)
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)

        return img, heatmap, centermap, img_path


    def __len__(self):
        return len(self.img_List)
