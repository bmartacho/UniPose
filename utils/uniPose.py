from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

import torch.nn.functional as F
import numpy as np
import torch
import math
import time
import cv2

def uniPose_kpts(maps, dataset, img_h = 368.0, img_w = 368.0):
    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    mapping = maps[0]


    if dataset == "LSP":
        center      = mapping[15]
        topLeft     = mapping[16]
        bottomLeft  = mapping[17]
        topRight    = mapping[18]
        bottomRight = mapping[19]


    elif dataset == "MPII":
        center      = mapping[17]
        topLeft     = mapping[18]
        bottomLeft  = mapping[19]
        topRight    = mapping[20]
        bottomRight = mapping[21]


    elif dataset == "PoseTrack":
        center      = mapping[18]
        topLeft     = mapping[19]
        bottomLeft  = mapping[20]
        topRight    = mapping[21]
        bottomRight = mapping[22]


    elif dataset == "NTID" or dataset == "NTID_small":
        center      = mapping[20]
        topLeft     = mapping[21]
        bottomLeft  = mapping[22]
        topRight    = mapping[23]
        bottomRight = mapping[24]


    threshold = 0


    center[center<threshold] = 0

    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(center, footprint=neighborhood)==center
    background = (center==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks    = local_max ^ eroded_background

    center = detected_peaks*center

    kpts_center = []
    for i in range(center.shape[0]):
        for j in range(center.shape[1]):
            if center[i,j] > 0:
                kpts_center.append([i,j])


    topLeft[topLeft<threshold] = 0

    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(topLeft, footprint=neighborhood)==topLeft
    background = (topLeft==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks    = local_max ^ eroded_background

    topLeft = detected_peaks*topLeft

    kpts_topLeft = []
    for i in range(topLeft.shape[0]):
        for j in range(topLeft.shape[1]):
            if topLeft[i,j] > 0:
                kpts_topLeft.append([i,j])


    bottomLeft[bottomLeft<threshold] = 0

    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(bottomLeft, footprint=neighborhood)==bottomLeft
    background = (bottomLeft==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks    = local_max ^ eroded_background

    bottomLeft = detected_peaks*bottomLeft

    kpts_bottomLeft = []
    for i in range(bottomLeft.shape[0]):
        for j in range(bottomLeft.shape[1]):
            if bottomLeft[i,j] > 0:
                kpts_bottomLeft.append([i,j])


    topRight[topRight<threshold] = 0

    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(topRight, footprint=neighborhood)==topRight
    background = (topRight==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks    = local_max ^ eroded_background

    topRight = detected_peaks*topRight

    kpts_topRight = []
    for i in range(topRight.shape[0]):
        for j in range(topRight.shape[1]):
            if topRight[i,j] > 0:
                kpts_topRight.append([i,j])


    bottomRight[bottomRight<threshold] = 0

    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(bottomRight, footprint=neighborhood)==bottomRight
    background = (bottomRight==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks    = local_max ^ eroded_background

    bottomRight = detected_peaks*bottomRight

    kpts_bottomRight = []
    for i in range(bottomRight.shape[0]):
        for j in range(bottomRight.shape[1]):
            if bottomRight[i,j] > 0:
                kpts_bottomRight.append([i,j])


    kpts = []

    # print(kpts_topLeft[0])

    for idx in range(len(kpts_center)):
        # y1, x1 = kpts_topLeft[idx][0], kpts_topLeft[idx][0][1]
        # y2, x2 = np.unravel_index(kpts_bottomRight[idx].argmax(), kpts_bottomRight[idx].shape)

        box = mapping[:,kpts_topLeft[idx][0]:kpts_bottomRight[idx][0],\
                        kpts_topLeft[idx][1]:kpts_bottomRight[idx][1]]

        # print(box.shape)

        for m in box[1:15]:
            h, w = np.unravel_index(m.argmax(), m.shape)
            x = int(w+kpts_topLeft[idx][1])
            y = int(h+kpts_topLeft[idx][0])
            kpts.append([idx,x,y])


        kpts.append([idx,kpts_center[idx][1],      kpts_center[idx][0]])
        kpts.append([idx,kpts_topLeft[idx][1],     kpts_topLeft[idx][0]])
        kpts.append([idx,kpts_bottomLeft[idx][1],  kpts_bottomLeft[idx][0]])
        kpts.append([idx,kpts_topRight[idx][1],    kpts_topRight[idx][0]])
        kpts.append([idx,kpts_bottomRight[idx][1], kpts_bottomRight[idx][0]])

    # kpts_center      = []
    # kpts_topLeft     = []
    # kpts_bottomLeft  = []
    # kpts_topRight    = []
    # kpts_bottomRight = []
    # for i in range(center.shape[0]):
    #     for j in range(center.shape[1]):
    #         if center[i,j] > 0:
    #             kpts_center.append([i,j])
    #         if kpts_topLeft[i,j] > 0:
    #             kpts_topLeft.append([i,j])
    #         if kpts_bottomLeft[i,j] > 0:
    #             kpts_bottomLeft.append([i,j])
    #         if kpts_topRight[i,j] > 0:
    #             kpts_topRight.append([i,j])
    #         if kpts_bottomRight[i,j] > 0:
    #             kpts_bottomRight.append([i,j])

    # print(kpts.shape)
    # print(1,kpts[0:mapping.shape[0]-1])
    # print(2,kpts[mapping.shape[0]:])

    # print(kpts)

    return kpts


def draw_paint(img_path, kpts, mapNumber, epoch, model_arch, dataset):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [255,0,0], \
              [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0]]

    if dataset   == "LSP":
        limbSeq = [[13, 12], [12, 9], [12, 8], [ 9, 10], [ 8, 7], [10, 11], [7,  6], [12, 3], [12, 2], [ 2, 1],\
                   [ 1, 0], [ 3, 4], [ 4, 5], [15, 16], [16,18], [17, 18], [15,17]]

    elif dataset == "MPII":
        limbSeq = [[0, 1], [ 1, 2], [ 2, 6], [ 5, 4], [ 4, 3], [ 3, 6], [ 6, 7], [ 7, 8], [ 8, 9], [7,12], \
                   [7,13], [12,11], [11,10], [13,14], [14,15], [17,18], [18,20], [19,20], [17,19]]

    elif dataset == "PoseTrack":
        limbSeq = [[16,14], [14,12], [17,15], [15,13], [12,13], [ 6,12], [ 7,13], [ 6, 7], [ 7, 8], [7, 9], \
                   [ 8,10], [ 9,11], [ 2, 3], [ 1, 2], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6], [ 5, 7]]

    print(len(kpts))

    im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))


    print(kpts)
    num_people = kpts[-1][0]+1
    num_kpts   = int(len(kpts)/num_people)

    print(num_people, num_kpts)

    # print(kpts)

    current = 0

    # draw lines
    for idx in range(num_people):
        person  = kpts[current:current+num_kpts]
        current = idx*num_kpts+num_kpts
        # print(current)
        print(person)

        for i in range(len(limbSeq)):

            cur_im = im.copy()
            limb = limbSeq[i]
            # [Y0, X0] = kpts[limb[0]]
            Y0 = person[limb[0]][1]
            X0 = person[limb[0]][2]
            # [Y1, X1] = kpts[limb[1]]
            Y1 = person[limb[1]][1]
            X1 = person[limb[1]][2]

            mX = np.mean([X0, X1])
            mY = np.mean([Y0, Y1])
            length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            if i < 12:
                cv2.fillConvexPoly(cur_im, polygon, colors[i])
            else:
                cv2.fillConvexPoly(cur_im, polygon, [0, 0, 255])
            im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)


            # print([Y0,X0], [Y1,X1])

            # quit()

    cv2.imwrite('samples/WASPpose/Pose/'+str(mapNumber)+'.png', im)