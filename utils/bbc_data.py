# -*-coding:UTF-8-*-
import os
import time
import scipy.io
import numpy as np
import glob
import torch
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import math
import utils.Mytransforms as Mytransforms
from torchvision import transforms


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def getBoundingBox(img, kpt, height, width, stride):
    x = []
    y = []

    for index in range(0,len(kpt)):
        if float(kpt[index][1]) >= 0 or float(kpt[index][0]) >= 0:
            x.append(float(kpt[index][1]))
            y.append(float(kpt[index][0]))

    if len(x) == 0 or len(y) == 0:
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

    else:
        x_min = int(max(min(x), 0))
        x_max = int(min(max(x), width))
        y_min = int(max(min(y), 0))
        y_max = int(min(max(y), height))

    # box = np.zeros(4)
    # box[0] = (x_min + x_max)/2
    # box[1] = (y_min + y_max)/2
    # box[2] =  x_max - x_min
    # box[3] =  y_max - y_min

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

    box = np.zeros((int(height/stride), int(width/stride), 5), dtype=np.float32)
    for i in range(5):
        # resize from 368 to 46
        x = int(coord[i][0]) * 1.0
        y = int(coord[i][1]) * 1.0
        heat_map = guassian_kernel(size_h=int(height/stride), size_w=int(width/stride), center_x=x, center_y=y, sigma=3)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        box[:, :, i] = heat_map

    return box


class BBC(data.Dataset):
    def __init__(self, root_dir, sigma, is_train, transform=None):
        self.width     = 800
        self.height    = 800
        self.transform = transform
        self.is_train  = is_train
        self.sigma     = sigma
        self.parts_num = 7
        self.seqTrain  = 5

        self.labels_dir  = root_dir + 'labels/'
        self.images_dir  = root_dir + 'BBCpose/'

        self.videosFolders = {}
        self.labelFiles    = {}
        self.imageFiles    = {}

        if is_train == "Train":
        	self.videos = [f for f in os.listdir(self.labels_dir+"train/imageFiles/")]
        	self.videos.sort()

        	for i in range(len(self.videos)):
        		self.labelFiles[i] = self.labels_dir+"train/joints/"+str(i+1)+"_"
        		self.imageFiles[i] = self.labels_dir+"train/imageFiles/"+str(i+1)+"_"


        elif is_train == "Val":
        	self.videos = [f for f in os.listdir(self.labels_dir+"val/imageFiles/")]
        	self.videos.sort()

        	for i in range(len(self.videos)):
        		self.labelFiles[i] = self.labels_dir+"val/joints/"+str(i+11)+"_"
        		self.imageFiles[i] = self.labels_dir+"val/imageFiles/"+str(i+11)+"_"


        elif is_train == "Test":
            self.videos = [f for f in os.listdir(self.labels_dir+"test/imageFiles/")]
            self.videos.sort()

            for i in range(len(self.videos)):
                self.labelFiles[i] = self.labels_dir+"test/joints/"+str(i+16)+"_"
                self.imageFiles[i] = self.labels_dir+"test/imageFiles/"+str(i+16)+"_"
       
       

        # print(self.labelFiles[0],self.labelFiles[1])
        # print(self.imageFiles[0],self.imageFiles[1])
       	


        self.img_List = []
        self.kps      = []
        self.centers  = []
        count = 0
        for idx in range(len(self.labelFiles)):
            frames        = scipy.io.loadmat(self.imageFiles[idx] + "imageFiles.mat")['imageFile']
            kpoints_List  = scipy.io.loadmat(self.labelFiles[idx] + "jointFiles.mat")['joints']

            if is_train == "Train":
            	for j in range(frames.shape[1]):
            		self.img_List.append(self.images_dir + str(idx+1) + "/" + str(int(frames[0,j])) + ".jpg")

            elif is_train == "Val":
            	for j in range(frames.shape[0]):
            		self.img_List.append(self.images_dir + str(idx+11) + "/" + str(int(frames[j])) + ".jpg")

            elif is_train == "Test":
                for j in range(frames.shape[0]):
                    self.img_List.append(self.images_dir + str(idx+16) + "/" + str(int(frames[j])) + ".jpg")

            if idx == 0:
            	self.kps = kpoints_List
            else:
            	self.kps = np.append(self.kps,kpoints_List,axis=0)

            # print(len(self.img_List), self.kps.shape)


        if is_train == "Train":
            print("Train      Images = " + str(len(self.img_List)))
        elif is_train == "Val":
            print("Validation Images = " + str(len(self.img_List)))
        elif is_train == "Test":
            print("Test       Images = " + str(len(self.img_List)))


    def __getitem__(self, index):
        im = cv2.imread(self.img_List[index])
        if im is None:
            print(self.img_List[index])
            im  = cv2.imread(self.img_List[index-1])
        img     = np.array(im,dtype=np.float32)
        kps     = self.kps[index]
        shift = [img.shape[1]-368, img.shape[0]-368]
        img = img[-368:,-368:,:]

        # print(kps)

        kps[:,0] = kps[:,0]-shift[0]
        kps[:,1] = kps[:,1]-shift[1]

        # print(kps)

        # print(self.img_List[index])
        # print(img.shape)

        center = {}

        center[0] = [img.shape[0]/2,img.shape[1]/2]

        # print(kps.shape, img.shape)

        # expand dataset
        # if self.is_train == "Train":
        #     img, kps, center = self.transform(img, kps, center)
        height, width, _ = img.shape


        kps[kps<0] = 0

        # limbsMap = getLimbs(img, kpt, height, width, 8, self.bodyParts, self.parts_num, 1)
        box      = getBoundingBox(img, kps, height, width, 8)


        heatmap = np.zeros((46, 46, int(len(kps)+1)), dtype=np.float32)
        for i in range(len(kps)):
            # resize from 368 to 46
            x = int(kps[i][0]) * 1.0 / 8
            y = int(kps[i][1]) * 1.0 / 8
            heat_map = guassian_kernel(size_h=368/8, size_w=368/8, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=184, center_y=184, sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        # limbsMap  = Mytransforms.to_tensor(limbsMap)
        box       = Mytransforms.to_tensor(box)

        return img, heatmap, centermap, self.img_List[index], 0, box



        return img, 0, 0, self.img_List[index], 0, 0


    def __len__(self):
        return len(self.img_List)

