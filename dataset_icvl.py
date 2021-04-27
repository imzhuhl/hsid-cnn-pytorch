import numpy as np
import os
import torch
import scipy.io
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import get_train_name, get_test_name


class TrainData(Dataset):
    def __init__(self, orig_dir, patch_size, my_transform):
        self.file_names = get_train_name()
        self.orig_dir = orig_dir
        self.orig_imgs = self.load_data()
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
        im = input_im[:, y1:y1+patch_size, x1:x1+patch_size]
        vol = input_vol[:, y1:y1+patch_size, x1:x1+patch_size]
        return im, vol

    def load_data(self):
        orig_imgs = []
        crop_size = 1024
        for name in self.file_names:
            file_path = os.path.join(self.orig_dir, name)
            orig_img = scipy.io.loadmat(file_path)['img']
            orig_img = np.transpose(orig_img, (2, 0, 1))  # HWC -> CHW
            orig_imgs.append(orig_img)
        orig_imgs = np.array(orig_imgs, dtype=np.float32)
        orig_imgs = np.reshape(orig_imgs, (-1, crop_size, crop_size))
        print('total training data:', len(orig_imgs))
        return orig_imgs


class TestData(Dataset):
    def __init__(self, orig_dir, noise_dir, scene_id=-1):
        """
        Args:
            scene_id: `-1` means all test images 
        """
        super(TestData, self).__init__()
        self.file_names = get_test_name()
        self.orig_dir = orig_dir
        self.noise_dir = noise_dir
        self.scene_id = scene_id
        self.orig_imgs, self.noise_imgs = self.load_data()

    def load_data(self):
        orig_imgs = []
        noise_imgs = []
        for i, name in enumerate(self.file_names):
            if self.scene_id == -1 or self.scene_id == i:
                file_path = os.path.join(self.orig_dir, name)
                orig_img = scipy.io.loadmat(file_path)['img']
                file_path = os.path.join(self.noise_dir, name)
                noise_img = scipy.io.loadmat(file_path)['img_n']
                orig_img = np.transpose(orig_img, (2, 0, 1))  # HWC -> CHW
                noise_img = np.transpose(noise_img, (2, 0, 1)) 
                orig_imgs.append(orig_img)
                noise_imgs.append(noise_img)
        orig_imgs = np.array(orig_imgs, dtype=np.float32)
        noise_imgs = np.array(noise_imgs, dtype=np.float32)
        orig_imgs = np.reshape(orig_imgs, (-1, 512, 512))
        noise_imgs = np.reshape(noise_imgs, (-1, 512, 512))
     
        return orig_imgs, noise_imgs

    def __getitem__(self, index):
        noise_im = self.noise_imgs[index]
        orig_im = self.orig_imgs[index]
        noise_im = noise_im[np.newaxis, :, :]
        orig_im = orig_im[np.newaxis, :, :]

        if index < 12:
            noise_vol = self.noise_imgs[0:24, :, :]
        elif index > 18:
            noise_vol = self.noise_imgs[7:31, :, :]
        else:
            noise_vol_1 = self.noise_imgs[index-12:index, :, :]
            noise_vol_2 = self.noise_imgs[index+1:index+13, :, :]
            noise_vol = np.concatenate((noise_vol_1, noise_vol_2), axis=0)

        noise_im = torch.from_numpy(noise_im).float()
        noise_vol = torch.from_numpy(noise_vol).float()
        orig_im = torch.from_numpy(orig_im).float()
        sample = {
            'input_im': noise_im,
            'input_vol': noise_vol,
            'target_im': orig_im,
        }
        return sample

    def __len__(self):
        return len(self.orig_imgs)


if __name__ == '__main__':
    # TODO test class TrainData
    # mytf = transforms.Compose([
    #     RandGaNoise(55),
    #     ToTensor(),
    # ])
    # dataset = TrainData('../dncnn-icvl/data/train/orig', 64, mytf)
    # for i, sample in enumerate(dataset):
    #     print()

    dataset = TestData('../dncnn-icvl-2/data/test/orig', '../dncnn-icvl-2/data/test/randga55', -1)
    for i, sample in enumerate(dataset):
        print()