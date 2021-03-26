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


class Penn_Action(data.Dataset):
    def __init__(self, root_dir, sigma, frame_memory, is_train, transform=None):
        self.width     = 368
        self.height    = 368
        self.transform = transform
        self.is_train  = is_train
        self.sigma     = sigma
        self.parts_num = 13
        self.seqTrain  = frame_memory

        self.root_dir     = root_dir
        self.label_dir    = root_dir + 'labels/'
        self.frame_dir    = root_dir + 'frames/' 
        self.train_dir    = root_dir + 'train/' 
        self.val_dir      = root_dir + 'val/' 

        if self.is_train is True:
            self.data_dir = root_dir + 'train/'
        else:
            self.data_dir = root_dir + 'val/'

        self.frames_data  = os.listdir(self.data_dir)


    def __getitem__(self, index):
        frames = self.frames_data[index]
        data   = np.load(os.path.join(self.data_dir, frames)).item()

        nframes    = data['nframes']    # 151
        framespath = data['framepath']
        dim        = data['dimensions'] # [360,480]
        x          = data['x']          # 151 * 13
        y          = data['y']          # 151 * 13
        visibility = data['visibility'] # 151 * 13

        img_paths    = []

        start_index = np.random.randint(0, nframes - 1 - self.seqTrain + 1)

        images     = torch.zeros(self.seqTrain, 3, self.height, self.width)  # [3,368,368]
        centermaps = torch.zeros(self.seqTrain, 1, self.height, self.width)  # [3,368,368]
#         boxes      = torch.zeros(self.seqTrain, 5, self.height, self.width)  # [3,368,368]
        label      = np.zeros((3, 13, self.seqTrain))
        lms        = np.zeros((1, 3, 13))
        kps        = np.zeros((13 + 5, 3))

        # build data set--------
        label_size = 368
        label_map  = torch.zeros(self.seqTrain, self.parts_num + 1, label_size, label_size)


        for i in range(self.seqTrain):
            # read image
            img_path = os.path.join('/home/bm3768/Desktop/Pose/'+framespath[34:],\
                                    '%06d' % (start_index + i + 1) + '.jpg')

            img_paths.append(img_path)
            img = np.array(Image.open(img_path), dtype=np.float32)  # Image


            img_path  = self.images_dir + variable['img_paths']
            
            # BBox was added to the labels by the authors to perform additional training and testing, as referred in the paper.
            # Intentionally left as comment since it is not part of the dataset.
#             bbox      = np.load(self.labels_dir + "BBOX/" + variable['img_paths'][:-4] + '.npy')

            # read label
            label[0, :, i] = x[start_index + i]
            label[1, :, i] = y[start_index + i]
            label[2, :, i] = visibility[start_index + i]  # 1 * 13
#             bbox[i, :]       = data['bbox'][start_index + i]  #


            # make the joints not in the figure vis=-1(Do not produce label)
            for part in range(0, 13):  # for each part
                if self.isNotOnPlane(label[0, part, i], label[1, part, i], dim[1], dim[0]):
                    label[2, part, i] = -1


            temp2      = label.transpose([2, 1, 0])
            kps[:13]   = temp2[0]

            center_x = int(label_size/2)
            center_y = int(label_size/2)


            kps[13] = [int((bbox[i,0]+bbox[i,2])/2),int((bbox[i,1]+bbox[i,3])/2),1]
            kps[14] = [bbox[i,0],bbox[i,1],1] 
            kps[15] = [bbox[i,0],bbox[i,3],1] 
            kps[16] = [bbox[i,2],bbox[i,1],1] 
            kps[17] = [bbox[i,2],bbox[i,3],1] 

            center   = [center_x, center_y]

            img, kps, center = self.transform(img, kps, center)

            box  = kps[-5:]
            kpts = kps[:13]


            label[:,:,i] = kpts.transpose(1,0)

            img = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)  # Image

            images[i, :, :, :] = transforms.ToTensor()(img)

            centermap = np.zeros((self.height, self.width, 1), dtype=np.float32)
            center_map = guassian_kernel(size_h=self.height, size_w=self.width, center_x=center[0], center_y=center[1], sigma=3)
            center_map[center_map > 1] = 1
            center_map[center_map < 0.0099] = 0
            centermap[:, :, 0] = center_map

            centermaps[i, :, :, :] = transforms.ToTensor()(centermap)

            heatmap = np.zeros((368, 368, 14), dtype=np.float32)
            for k in range(13):
                # resize from 368 to 46
                xk = int(kpts[k][0])
                yk = int(kpts[k][1])
                heat_map = guassian_kernel(size_h=368, size_w=368, center_x=xk, center_y=yk, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmap[:, :, k+1] = heat_map

#             box_heatmap = np.zeros((368, 368, 5), dtype=np.float32)
#             for k in range(5):
#                 # resize from 368 to 46
#                 xk = int(box[k][0])
#                 yk = int(box[k][1])
#                 heat_map = guassian_kernel(size_h=368, size_w=368, center_x=xk, center_y=yk, sigma=self.sigma)
#                 heat_map[heat_map > 1] = 1
#                 heat_map[heat_map < 0.0099] = 0
#                 box_heatmap[:, :, k] = heat_map

            heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background      

            label_map[i] = transforms.ToTensor()(heatmap)
#             boxes[i]     = transforms.ToTensor()(box_heatmap)


        for i in range(self.seqTrain):
            images[i] = Mytransforms.normalize(images[i], [128.0, 128.0, 128.0],[256.0, 256.0, 256.0])


        label_map[i] = transforms.ToTensor()(heatmap)


        return images, label_map, centermaps, img_paths#, 0, boxes


    def isNotOnPlane(self, x, y, width, height):
        notOn = x < 0.001 or y < 0.001 or x > width or y > height
        return notOn


    def __len__(self):
        return len(self.frames_data)

