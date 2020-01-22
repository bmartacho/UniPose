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
        heat_map = guassian_kernel(size_h=height/stride, size_w=width/stride, center_x=x, center_y=y, sigma=3)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        box[:, :, i] = heat_map

    return box

def getLimbs(img, kpt, height, width, stride, bodyParts, thickness, sigma):
    nParts    = len(bodyParts)
    limb_maps = np.zeros((nParts, height//stride, width//stride), dtype=np.float)

    #           Head; R. Shoulder; Left Shoulder; Right Biceps; Left Biceps; Right Forearm;
    #           Left Forearm; Torso; Hip; Right Thigh, Left Thigh, Right calf; Left Calf

    for idx in range(nParts):
        if idx == 7:
            keya = [int(kpt[bodyParts[idx][0]][0]/stride), int(kpt[bodyParts[idx][0]][1]/stride)]
            keyb = [int((kpt[2][0]+kpt[3][0])/(2*stride)), int((kpt[2][1]+kpt[3][1])/(2*stride))]

        else:
            keya = [int(kpt[bodyParts[idx][0]][0]/stride), int(kpt[bodyParts[idx][0]][1]/stride)]
            keyb = [int(kpt[bodyParts[idx][1]][0]/stride), int(kpt[bodyParts[idx][1]][1]/stride)]

        vector        = [keya[0] - keyb[0], keya[1] - keyb[1]]
        vector        = [keya[0] - keyb[0], keya[1] - keyb[1]]
        normalization = (vector[0]*vector[0] + vector[1]*vector[1]) ** 0.5

        if normalization != 0:
            unit_vector   = [vector[0]/normalization, vector[1]/normalization]

            x_min = int(max(min(keya[1], keyb[1]), 0))
            x_max = int(min(max(keya[1], keyb[1]), limb_maps.shape[2]))
            y_min = int(max(min(keya[0], keyb[0]), 0))
            y_max = int(min(max(keya[0], keyb[0]), limb_maps.shape[1]))

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    xca = x - keya[1]
                    yca = y - keya[0]
                    xcb = x - keyb[1]
                    ycb = y - keyb[0]
                    #d   = math.fabs(xca*unit_vector[0]-yca*unit_vector[1])
                    d   = math.fabs((keyb[0]-keya[0])*x - (keyb[1]-keya[1])*y + keyb[1]*keya[0] - keya[1]*keyb[0])/ \
                                 ((keyb[0]-keya[0])*(keyb[0]-keya[0]) + (keyb[1]-keya[1])*(keyb[1]-keya[1])) ** 0.5

                    limb_maps[idx,x,y] = np.exp(-(d*d) / 2.0 / (sigma*sigma))

                    if limb_maps[idx,x,y] > 1:
                        limb_maps[idx,x,y] = 1
                    elif limb_maps[idx,x,y] < 0.0099:
                        limb_maps[idx,x,y] = 0

    return limb_maps.transpose(1,2,0)


class NTID(data.Dataset):
    def __init__(self, root_dir, sigma, is_train, transform=None):
        self.width     = 800
        self.height    = 800
        self.transform = transform
        self.is_train  = is_train
        self.sigma     = sigma
        self.parts_num = 19 #25
        self.seqTrain  = 5

        #root_dir = "dataset/NTID/"

        self.labels_dir  = root_dir + 'labels/'
        self.images_dir  = root_dir + 'images/'

        self.videosFolders = {}
        self.labelFiles    = {}
        self.imageFiles    = {}

        self.videosFolders = [f for f in os.listdir(self.labels_dir) if not os.path.isfile(os.path.join(self.labels_dir,f))]
        self.videosFolders.sort()

        count = 0
        for j in range(len(self.videosFolders)):
            self.labelPath = self.labels_dir + self.videosFolders[j]+"/"

            self.labelFiles[count] = [f for f in os.listdir(self.labelPath)]

            self.imagePath = self.images_dir  + self.videosFolders[j]+"/"

            self.imageFiles[count] = [f for f in os.listdir(self.imagePath)]
            self.imageFiles[count].sort()

            count = count + 1

        self.img_List = []
        self.kps      = []
        self.centers  = []
        for idx in range(len(self.videosFolders)):
            img_List      = np.array(glob.glob(self.images_dir + self.videosFolders[idx] + "/*.png"))
            self.img_List = np.append(self.img_List,img_List, axis=0)

            kpoints_List  = scipy.io.loadmat(self.labels_dir + self.videosFolders[idx]+"/"+\
                                             self.labelFiles[idx][0])['arr']

            kpoints_List  = np.delete(kpoints_List,[26,27,28,29,30,31,34,35,36,37,38,39],axis=1)

            if idx == 0:
                self.kps  = kpoints_List
            else:
                self.kps  = np.append(self.kps,kpoints_List, axis=0)

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

        center = {}

        center[0] = [img.shape[0]/2,img.shape[1]/2]

        prefix_length = self.img_List[index].rfind('images')
        sufix_start   = self.img_List[index].rfind('/')
        frame_number  = int(self.img_List[index][sufix_start+1:-4])
        label_file = self.img_List[index][:prefix_length]+"labels"+\
                     self.img_List[index][prefix_length+6:sufix_start+1]+\
                     self.img_List[index][prefix_length+11:sufix_start]+".mat"


        if scipy.io.loadmat(label_file)['arr'][frame_number] is None:
            print(self.img_List[index])
            kps2 = scipy.io.loadmat(label_file)['arr'][frame_number-1]
            im  = cv2.imread(self.img_List[index-1])

        else:
            kps2 = scipy.io.loadmat(label_file)['arr'][frame_number]

        # Remove Knee, ankle and foot detections (not visible in the image)
        kps2 = np.delete(kps2,[26,27,28,29,30,31,34,35,36,37,38,39],axis=0)

        kpt = np.zeros((self.parts_num,2))
        # kpt2 = np.zeros((self.parts_num,2))
        for i in range(0,self.parts_num):
            kpt[i, 0] = kps2[2*i]
            kpt[i, 1] = kps2[2*i+1]

            # kpt2[i, 0] = kps2[2*i]
            # kpt2[i, 1] = kps2[2*i+1]



        # expand dataset
        # if self.is_train == "Train":
        #     img, kpt, center = self.transform(img, kpt, center)
        height, width, _ = img.shape


        kpt[kpt<0] = 0

        # limbsMap = getLimbs(img, kpt, height, width, 8, self.bodyParts, self.parts_num, 1)
        box      = getBoundingBox(img, kpt, height, width, 8)


        heatmap = np.zeros((int(height/8), int(width/8), int(len(kpt)+1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / 8
            y = int(kpt[i][1]) * 1.0 / 8
            heat_map = guassian_kernel(size_h=height / 8, size_w=width / 8, center_x=x, center_y=y, sigma=self.sigma)
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



        # return img, 0, 0, self.img_List[index], 0, 0


    def __len__(self):
        return len(self.img_List)

