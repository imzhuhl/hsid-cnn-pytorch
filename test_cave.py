import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
from torch import optim
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import scipy.io

from network import HSID
from utils import calc_psnr

CUDA_ID = 7
DEVICE = torch.device(f'cuda:{CUDA_ID}')

TEST_CFG = {
    "randga": {
        "test_dataset_path": "../dncnn-2/cave/randga25/test.npz",
        "scene_id": 0,
        "result_dir": "results/cave/randga25",
        "model_path": "saved_models/icvl/randga55/pretrained/experiment_0/models/epoch_92",
    },
}


class CaveDataset(Dataset):
    def __init__(self, npz_path, scene_id=0):
        super(CaveDataset, self).__init__()
        data = np.load(npz_path)

        clean_imgs = data['clean_img'][scene_id]
        noise_imgs = data['noise_img'][scene_id]

        # nhwc -> nchw
        clean_imgs = np.transpose(clean_imgs, (2, 0, 1))
        noise_imgs = np.transpose(noise_imgs, (2, 0, 1))

        # # convert to (n*c, h, w)
        # self.clean_imgs = np.reshape(clean_imgs, (-1, 512, 512))
        # self.noise_imgs = np.reshape(noise_imgs, (-1, 512, 512))
        self.clean_imgs = clean_imgs
        self.noise_imgs = noise_imgs
        assert len(self.clean_imgs) == len(self.noise_imgs)

        # self.pre_transform = pre_transform
        # self.inputs_transform = inputs_transform
    
    def __getitem__(self, index):
        noise_im = self.noise_imgs[index]
        clean_im = self.clean_imgs[index]
        noise_im = noise_im[np.newaxis, :, :]
        clean_im = clean_im[np.newaxis, :, :]
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
        clean_im = torch.from_numpy(clean_im).float()
        sample = {
            'input_im': noise_im,
            'input_vol': noise_vol,
            'target_im': clean_im,
        }
        return sample
        
    def __len__(self):
        return len(self.clean_imgs)


class Tester:
    def __init__(self) -> None:
        test_cfg = TEST_CFG['randga']
        self.scene_id = test_cfg['scene_id']
        saved_model_path = test_cfg['model_path']
        test_data_path = test_cfg['test_dataset_path']
        result_dir = test_cfg['result_dir']
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        # load model
        model = HSID(24)
        model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model'], strict=True)
        model = model.to(DEVICE)
        # load test dataset
        test_dataset = CaveDataset(test_data_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        denoise_img, clean_img, psnr = self.test(model, test_loader)
        print(f'{psnr:.2f}')
        scipy.io.savemat(os.path.join(result_dir, 'hsidcnn.mat'), {'img': clean_img, 'img_n': denoise_img})
    
    def test(self, model, test_loader):
        denoise_imgs = []
        clean_imgs = []
        our_psnr = 0.0
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                input_im, input_vol, target_im = sample['input_im'].to(DEVICE), sample['input_vol'].to(DEVICE), sample['target_im'].to(DEVICE)
                noise_res = model(input_im, input_vol)
                denoise_img = input_im - noise_res

                denoise_img = denoise_img.detach().cpu().numpy()
                target_im = target_im.detach().cpu().numpy()
                denoise_img = np.squeeze(denoise_img)
                target_im = np.squeeze(target_im)
                our_psnr += calc_psnr(target_im, denoise_img)
                denoise_imgs.append(denoise_img)
                clean_imgs.append(target_im)
        denoise_imgs = np.array(denoise_imgs, dtype=np.float32)
        clean_imgs = np.array(clean_imgs, dtype=np.float32)
        denoise_imgs = np.transpose(denoise_imgs, (1, 2, 0))
        clean_imgs = np.transpose(clean_imgs, (1, 2, 0))
        return denoise_imgs, clean_imgs, our_psnr / len(test_loader)


if __name__ == '__main__':
    # CaveDataset('../dncnn-2/cave/randga75/test.npz')
    Tester()
