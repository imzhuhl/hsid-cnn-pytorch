import numpy as np
import os
import torch
import scipy.io
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainData(Dataset):
    def __init__(self, orig_dir, patch_size, my_transform):
        orig_imgs = scipy.io.loadmat(orig_dir)['img']
        self.orig_imgs = np.transpose(orig_imgs, (2, 0, 1))  # HWC -> CHW
        self.my_transform = my_transform
        self.patch_size = patch_size

    def __getitem__(self, index):
        clean_im = self.orig_imgs[index]
        clean_im = clean_im[np.newaxis, :, :]  #(1, h, w)
        if index < 12:
            clean_vol = self.orig_imgs[0:24, :, :]
        elif index > 18:
            clean_vol = self.orig_imgs[7:31, :, :]
        else:
            clean_vol_1 = self.orig_imgs[index-12:index, :, :]
            clean_vol_2 = self.orig_imgs[index+1:index+13, :, :]
            clean_vol = np.concatenate((clean_vol_1, clean_vol_2), axis=0)
        input_im, input_vol = self.rand_crop(clean_im, clean_vol, self.patch_size)
        target_im = input_im.copy()
        sample = {
            'input_im': input_im,
            'input_vol': input_vol,
            'target_im': target_im,
        }
        sample = self.my_transform(sample)
        return sample

    def __len__(self):
        return len(self.orig_imgs)

    def rand_crop(self, input_im, input_vol, patch_size):
        _, y,x = input_im.shape
        x1 = random.randint(0, x - patch_size)
        y1 = random.randint(0, y - patch_size)
        im = input_im[:, y1:y1+patch_size, x1:x1+patch_size].copy()
        vol = input_vol[:, y1:y1+patch_size, x1:x1+patch_size].copy()
        return im, vol


if __name__ == '__main__':
    from utils import RandGaNoise, ToTensor
    mytf = transforms.Compose([
        RandGaNoise(50),
        ToTensor(),
    ])
    dataset = TrainData('./data/dc/dc_norm.mat', 20, mytf)
    for i, sample in enumerate(dataset):
        print()
