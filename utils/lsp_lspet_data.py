# -*-coding:UTF-8-*-
import os
import scipy.io
import numpy as np
import glob
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import math
import utils.Mytransforms as Mytransforms
#from utils.utils import PAF as PAF


def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    image_arr = np.array(glob.glob(os.path.join(root_dir, 'images/*.jpg')))
    image_nums_arr = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_arr])
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    return sorted_image_arr.tolist()

def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'lsp' or 'lspet'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    # lspnet (14,3,10000)
    if mode == 'lspet':
        lms = mat_arr.transpose([2, 1, 0])
        kpts = mat_arr.transpose([2, 0, 1]).tolist()
    # lsp (3,14,2000)
    if mode == 'lsp':
        mat_arr[2] = np.logical_not(mat_arr[2])
        lms = mat_arr.transpose([2, 0, 1])
        kpts = mat_arr.transpose([2, 1, 0]).tolist()

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]
        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return kpts, centers, scales


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


def getLimbs(img, kpt, height, width, stride, bodyParts, thickness, sigma):
    """
         0 = Right Ankle
         1 = Right Knee
         2 = Right Hip
         3 = Left  Hip
         4 = Left  Knee
         5 = Left  Ankle
         6 = Right Wrist
         7 = Right Elbow
         8 = Right Shoulder
         9 = Left  Shoulder
        10 = Left  Elbow
        11 = Left  Wrist
        12 = Neck
        13 = Head  Top
    """
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


class LSP_Data(data.Dataset):
    """
         0 = Right Ankle
         1 = Right Knee
         2 = Right Hip
         3 = Left  Hip
         4 = Left  Knee
         5 = Left  Ankle
         6 = Right Wrist
         7 = Right Elbow
         8 = Right Shoulder
         9 = Left  Shoulder
        10 = Left  Elbow
        11 = Left  Wrist
        12 = Neck
        13 = Head  Top
    """

    def __init__(self, mode, root_dir, sigma, stride, transformer=None):

        self.img_list    = read_data_file(root_dir)
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        self.stride      = stride
        self.transformer = transformer
        self.sigma       = sigma
        self.bodyParts   = [[13, 12], [12, 9], [12, 8], [8, 7], [9, 10], [7, 6], [10, 11], [12, 3], [2, 3], [2, 1], [1, 0], [3, 4], [4, 5]]


    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)

        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        img, kpt, center = self.transformer(img, kpt, center, scale)
        height, width, _ = img.shape
        # limbsMap = getLimbs(img, kpt, height, width, self.stride, self.bodyParts, 25, 1)

        box = getBoundingBox(img, self.kpt_list[index], height, width, self.stride)

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

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        # limbsMap  = Mytransforms.to_tensor(limbsMap)
        box       = Mytransforms.to_tensor(box)

        return img, heatmap, centermap, img_path, 0, box


    def __len__(self):
        return len(self.img_list)