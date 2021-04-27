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
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import scipy.io

from network import HSID
from dataset_icvl import TestData
from utils import calc_psnr, get_test_name

CUDA_ID = 7
DEVICE = torch.device(f'cuda:{CUDA_ID}')

TEST_CFG = {
    "randga_icvl": {
        "test_orig_dir": "../dncnn-icvl-2/data/test/orig",
        "test_noise_dir": "../dncnn-icvl-2/data/test/randga95",
        "result_dir": "results/icvl/randga95/",
        "model_path": "saved_models/icvl/randga95/pretrained/experiment_0/models/epoch_90",
    },

}

class Tester:
    def __init__(self, cfg_name, scene_id=-1) -> None:
        test_cfg = TEST_CFG[cfg_name]
        saved_model_path = test_cfg['model_path']
        print(f'Load weights: {saved_model_path}')
        model = HSID(24)
        model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model'], strict=True)
        model = model.to(DEVICE)

        self.file_names = get_test_name()
        self.result_dir = test_cfg['result_dir']
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        
        for i, name in enumerate(self.file_names):
            if scene_id != -1 and scene_id != i:
                continue
            test_dataset = TestData(test_cfg['test_orig_dir'], test_cfg['test_noise_dir'], i)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            denoise_img, psnr = self.test(model, test_loader)
            print(f'{name}: {psnr:.2f}')
            scipy.io.savemat(os.path.join(self.result_dir, self.file_names[i]), {'img_n': denoise_img})

    def test(self, model, test_loader):
        imgs = []
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
                imgs.append(denoise_img)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = np.transpose(imgs, (1, 2, 0))
        return imgs, our_psnr / len(test_loader)


if __name__ == '__main__':
    # for i in range(0, 50):
    #     Tester('randga95', i)
    Tester('randga_icvl', 1)

