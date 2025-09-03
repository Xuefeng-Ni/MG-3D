import json

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import os
import SimpleITK as sitk
from tqdm import tqdm
from monai import data, transforms
from monai.transforms import *

from PIL import Image
import matplotlib.pyplot as plt


def norm(img):
    min, max = img.min(), img.max()
    img = (img - min)/(max - min + 1e-6)
    return img


def pre_process(img, min=-1000, max=500):
    output = img.copy()
    output[img < min] = min
    output[img > max] = max
    output = (output - min)/(max - min)

    return output


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    return cmap


def read(img, transpose=False):
    img = sitk.ReadImage(img)
    direction = img.GetDirection()
    origin = img.GetOrigin()
    Spacing = img.GetSpacing()

    img = sitk.GetArrayFromImage(img)
    if transpose:
        img = img.transpose(1, 2, 0)

    return img, direction, origin, Spacing


def vis():
    path = '/home/xuefeng/Covid19_20/exps/fold0_swin3d/pred_fold0/'
    img_path = path
    label_path = path
    pred_path = path

    ls = os.listdir(pred_path)
    ls = [x for x in ls if (('label' not in x) and ('csv' not in x) and ('png' not in x))]
    save_path = path+'png/'

    import random
    # random.shuffle(ls)
    ls.sort()

    for i in ls:
        print(i)
        img = read(os.path.join(img_path, i), True)[0]
        lab = read(os.path.join(label_path, i[:-10]+'_seg_label.nii.gz'), True)[0].astype(np.uint8)
        pred = read(os.path.join(pred_path, 'pred_'+i[:-10]+'_seg_label.nii.gz'), True)[0].astype(np.uint8)
        print(img.shape, lab.shape)

        cls_set = list(np.unique(lab))
        print(cls_set)

        h, w, c = img.shape
        cmap = color_map()

        k = 0
        for j in range(c):
            im = img[:, :, j]
            la = lab[:, :, j]
            pre = pred[:, :, j]
            
            interaction = np.logical_and(la, pre).astype(int)
            miss_seg = la - interaction
            over_seg = pre - interaction

            cls_set = list(np.unique(la))

            if len(cls_set) > 1:

                color_seg = np.zeros((pre.shape[0], pre.shape[1], 3), dtype=np.uint8)
                color_seg[pre == 1, :] = [0, 128, 0]
                color_seg[over_seg == 1, :] = [128, 0, 0]
                color_seg[miss_seg == 1, :] = [128, 128, 0]
                
                im = (255 * im).astype(np.uint8)
                
                color = np.repeat(np.expand_dims(im, axis=2), 3, axis=2) * (1 - 0.6) + color_seg * 0.6
                color = color.astype(np.uint8)
                color = Image.fromarray(color.astype(np.uint8), mode='RGB')
                
                im = Image.fromarray(im)

                la = Image.fromarray(la.astype(np.uint8), mode='P')
                la.putpalette(cmap)

                pre = Image.fromarray(pre.astype(np.uint8), mode='P')
                pre.putpalette(cmap)

                la = blend(im, la)
                pre = blend(im, pre)

                fig, axs = plt.subplots(1, 3, figsize=(16, 5))
                axs[0].imshow(im, cmap='gray')
                axs[0].axis("off")

                axs[1].imshow(la)
                axs[1].axis("off")

                axs[2].imshow(pre)
                axs[2].axis("off")

                plt.tight_layout()
                plt.show()
                plt.close()

                name = k
                save_case_path = os.path.join(save_path, i[:-7]+'_'+str(name))
                k += 1
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                la.save(os.path.join(save_path, 'label_'+i[:-7]+'_'+str(name)+'.png'))
                pre.save(os.path.join(save_path, 'pred_'+i[:-7]+'_'+str(name)+'.png'))
                color.save(os.path.join(save_path, 'color_'+i[:-7]+'_'+str(name)+'.png'))


def blend(img, lab):
    img = img.convert('RGB')
    lab = lab.convert('RGB')
    ble = Image.blend(img, lab, 0.6)
    return ble


if __name__ == '__main__':
    vis()

