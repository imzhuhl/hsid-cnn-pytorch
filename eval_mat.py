import os
import numpy as np
import scipy.io

from skimage.metrics import structural_similarity

from utils import calc_psnr, get_test_name


def eval_icvl():
    orig_root = '../dncnn-icvl-2/data/test/orig'
    result_root = './results/icvl/randga95'
    file_names = get_test_name()

    scene_id = 0
    img = scipy.io.loadmat(os.path.join(orig_root, file_names[scene_id]))['img']
    img_n = scipy.io.loadmat(os.path.join(result_root, file_names[scene_id]))['img_n']
    _, _, c = img.shape
    
    print('{:30}{:<10}{:<10}{:<10}'.format('Scene', 'PSNR', 'SSIM', 'ERGAS'))

    psnr = 0.0
    ssim = 0.0
    ergas = 0.0
    for i in range(c):
        psnr += calc_psnr(img[:, :, i], img_n[:, :, i])
    
    psnr = psnr / c
    ssim = structural_similarity(img, img_n, win_size=11, data_range=1.0, multichannel=True, gaussian_weights=True)

    print("{:30}{:<10.2f}{:<10.4f}{:<10.2}".format(file_names[scene_id], psnr, ssim, ergas))



if __name__ == '__main__':
    eval_icvl()