# from DDPM_Net import Model
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os, itertools
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse
import cv2
from diffusers import UNet2DModel
from diffusion_diffusers_lzj import GaussianDiffusionTrainer, GaussianDiffusionSampler, DDIMSampler
# from DDPM_GaussianDiffusion import GaussianDiffusion, get_beta_schedule


gpu = 0
device, init_flag = core_lzj.cuda_init(gpu)
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
# parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
# parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
# parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
# parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=8000, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
# parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

step_size = 1024
img_size = 1024
img_step = int(img_size/step_size)
batch_size = 1
T = 1000
# t = 10




# core_lzj.check_folder_existence(save_dir)


transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((re_im_size, re_im_size)),
    # transforms.RandomCrop(crop_im_size, padding=0),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=0.5, std=0.5)
    # transforms.Normalize(mean=(0.5), std=(0.5))
])

transform_to_image = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((re_im_size_2, re_im_size_2)),
    # transforms.RandomCrop(crop_im_size, padding=0),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),

    # transforms.Normalize(mean=0.5, std=0.5)
    transforms.Normalize(mean=(-1), std=(2)),
    # transforms.ToPILImage(),
    # transforms.Resize((toimg_size, toimg_size))
])


def image2tensor(img):
    img[img > 4095] = 4095
    img1 = transform(Image.fromarray(img)).unsqueeze(0)
    img2 = img1.to(torch.float)
    img3 = img2 / 4095
    img4 = img3 * 2 - 1

    return img4


def image2tensor16bit(img):
    img1 = np.array(img, dtype=np.float32)
    img2 = torch.tensor(img1)
    img3 = img2.unsqueeze(0).unsqueeze(0) / 65535
    img4 = img3 * 2 - 1

    return img4


def image2show(img):
    img[img > 4095] = 4095
    img1 = img.astype(np.float32)
    img2 = img1 / 4095 * 65535
    img3 = img2.astype(np.uint16)
    img4 = Image.fromarray(img3)

    return img4


def image2show8bit(img):
    img[img > 4095] = 4095
    img1 = img.astype(np.float32)
    img2 = img1 / 4095 * 255
    img3 = img2.astype(np.uint8)
    img4 = Image.fromarray(img3)

    return img4

def image2show16bit(img):
    img1 = img.astype(np.float32)
    img2 = img1
    img3 = img2.astype(np.uint16)
    img4 = Image.fromarray(img3)

    return img4


def output2show16bit(img):
    img1 = img[0][0]
    img1[img1 < -1] = -1
    img1[img1 > 1] = 1
    img2 = ((img1 + 1) / 2 * 65535).astype(np.uint16)
    img3 = Image.fromarray(img2)

    return img3


def output2show8bit(img):
    img1 = img[0][0]
    img2 = ((img1 + 1) / 2 * 255)
    img2[img2 < 0] = 0
    img2[img2 > 255] = 255
    img3 = img2.astype(np.uint8)
    img4 = Image.fromarray(img3)

    return img4


def tensor2show8bit(img):
    img1 = img[0][0].detach().cpu().numpy()
    img2 = ((img1 + 1) / 2 * 255)
    # img2[img2 < 0] = 0
    # img2[img2 > 255] = 255
    img3 = img2.astype(np.uint8)
    img4 = Image.fromarray(img3)

    return img4


def tensor2show16bit(img):
    img1 = img[0][0].detach().cpu().numpy()
    img2 = ((img1 + 1) / 2 * 65535)
    # img2[img2 < 0] = 0
    # img2[img2 > 255] = 255
    img3 = img2.astype(np.uint16)
    img4 = Image.fromarray(img3)

    return img4


select = 0
if select == 0:
    img_dir = 'human stacks/G3-2_ave'
    # net_path = core_lzj.get_file()
    net_path = 'time_20241128001618_human brain size1024 T1000 v2 poisson/human brain size1024 T1000 v2 poisson_epochs_100.pkl'
else:
    save_dir = []
    net_path = []
    net2_path = []
    core_lzj.exit_program()


def get_matrix(net, sampler):
    net.eval()
    torch.set_grad_enabled(False)
    target = []
    # test = cv2.imread('denoise test/140.tif', -1)
    img_list = core_lzj.each_img_specify(img_dir, '.tif')
    img_list.sort()
    diff_steps = np.around(np.square(np.append(np.ones(10)*np.sqrt(10), np.linspace(np.sqrt(10),np.sqrt(120),390))))
    save_path = img_dir + '_size1024_net100 10-120 10 poisson square'
    core_lzj.check_folder_existence(save_path)
    # img = Image.open('9-1-256/9-1062.tif')
    # x0 = image2tensor16bit(img).to(device)
    # x0 = torch.randn(1, 1, img_size, img_size).to(device)
    # img_noise = output2show16bit(x0.detach().cpu().numpy())
    # img_noise.save(os.path.join(save_dir, 'noise.tif'))
    for step, img_path in enumerate(img_list):
        img_name = os.path.basename(img_path).split('.')[0]
        index = diff_steps[step]
        # save_img_path = os.path.join(save_path, img_name)
        # core_lzj.check_folder_existence(save_img_path)
        img = Image.open(img_path)
        img = img.resize((1024, 1024))
        x0 = image2tensor16bit(img).to(device)
        print('step:', step, ', ', 't:', index)
        # x0 = torch.randn(1, 1, img_size, img_size).to(device)
        # x0_pred = sampler(x0).detach().cpu().numpy()
        x0_pred = sampler(x0, index, 10, method="interval", eta=1).detach().cpu().numpy()
        img_denoise = output2show16bit(x0_pred)
        img_denoise.save(os.path.join(save_path, img_name + '_' + index.__str__() + '.tif'))
        del x0_pred

    return target


if __name__ == '__main__':

        # net2_path = 'united/low to high net.pkl'
    # elif select == 1:
    #     img_dir = core_lzj.get_file()
    #     net_path = core_lzj.get_file()
    # elif select == 2:
    #     img_dir = core_lzj.get_file()
    #     net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    # elif select == 3:
    #     img_dir = 'check/027h-2_18_21'
    #     net_path = core_lzj.get_file()

    model = UNet2DModel(
        sample_size=1024,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)
    sampler = DDIMSampler(model, 1e-4, 0.02, T).to(device)
    model.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['net_model'])
    # betas = get_beta_schedule('linear',
    #                           beta_start=0.0001,
    #                           beta_end=0.006,
    #                           num_diffusion_timesteps=T)
    #
    # betas = torch.from_numpy(betas).float().to(device)
    #
    #
    # model = Model(net_img_size).to(device)
    # model.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])
    # model.eval()
    #
    #
    # sample = GaussianDiffusion(betas=betas, device=device)
    target = get_matrix(net=model, sampler=sampler)

