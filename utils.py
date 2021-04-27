import math
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import random


def get_test_name():
    file = './ICVL_test_gauss.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def get_train_name():
    file = './ICVL_train.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


# def rand_crop(img, crop_size):
#     _, y,x = img.shape
#     x1 = random.randint(0, x - crop_size)
#     y1 = random.randint(0, y - crop_size)
#     return img[:, y1:y1+crop_size, x1:x1+crop_size]


class RandGaNoise(object):
    def __init__(self, sigma):
        self.sigma_ratio = sigma / 255.        
    
    def __call__(self, sample):
        im = sample['input_im']
        stddev_random = self.sigma_ratio * np.random.rand(1)  # 范围 stddev * (0 ~ 1)
        noise = np.random.randn(*im.shape) * stddev_random
        sample['input_im'] = im+noise

        vol = sample['input_vol']
        c, _, _ = vol.shape
        noise = np.random.randn(*vol.shape)
        for i in range(c):
            stddev_random = self.sigma_ratio * np.random.rand(1)  # 范围 stddev * (0 ~ 1)
            noise[i] = noise[i] * stddev_random
        sample['input_vol'] = vol + noise

        return sample


class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample = {
            'input_im': torch.from_numpy(sample['input_im']).float(),
            'input_vol': torch.from_numpy(sample['input_vol']).float(),
            'target_im': torch.from_numpy(sample['target_im']).float(),
        }
        return sample


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def save_train(path, model, optimizer, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if epoch is not None:
        state['epoch'] = epoch
    # `_use_new_zipfile...=False` support pytorch version < 1.6
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))


def init_exps(exp_root_dir):
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
    all_exps = glob.glob(f'{exp_root_dir}/experiment*')

    cur_exp_id = None
    if len(all_exps) == 0:
        cur_exp_id = 0
    else:
        exp_ids = [int(os.path.basename(s).split('_')[1]) for s in all_exps]
        exp_ids.sort()
        cur_exp_id = exp_ids[-1] + 1
        
    log_dir = f'{exp_root_dir}/experiment_{cur_exp_id}'
    os.makedirs(log_dir)

    return log_dir


def calc_psnr(im, recon, verbose=False):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    # mse = (np.linalg.norm(im-recon, ord=2) ** 2) / np.prod(im.shape)
    mse = np.sum((im - recon)**2) / np.prod(im.shape)
    # MAX = 1.0  #np.max(im)
    # max_val = np.max(im)
    max_val = 1.0
    psnr = 10 * np.log10(max_val ** 2 / mse)
    if verbose:
        print('PSNR %f'.format(psnr))
    return psnr


if __name__ == '__main__':
    pass
