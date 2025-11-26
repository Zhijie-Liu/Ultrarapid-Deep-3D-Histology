from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
# from model_virtual import Generator, Discriminator
# from networks import ResnetGenerator, PixelDiscriminator
from unet3d_12g.unet3d import CNN3D_1um
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage import io


gpu = 0
device, init_flag = core_lzj.cuda_init(gpu)


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
# parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
# parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
# parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
# parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=600, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=10, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

img_size = [50, 120, 120]
space_size = [10, 24, 24]
step = [int(img / space) for img, space in zip(img_size, space_size)]

print('step:', step)


normalize = transforms.Normalize((0.5), (0.5))
anti_normalize = transforms.Normalize((-1), (2))


def output2show16bit(img):
    img1 = anti_normalize(img)[0][0].detach().cpu().numpy()
    # img1[img1 < 0] = 0
    # img1[img1 > 1] = 1
    img2 = (img1 * 65535).astype(np.uint16)

    return img2


def output2numpy(img):
    img1 = anti_normalize(img)[0][0].detach().cpu().numpy()
    # img1[img1 < 0] = 0
    # img1[img1 > 1] = 1
    return img1

def get_volume(model, img_mat, img_volume):
    target = np.zeros(shape=(img_mat[0] * space_size[0], img_mat[1] * space_size[1], img_mat[2] * space_size[2]), dtype=np.float32)
    adjust = np.zeros(shape=(img_mat[0] * space_size[0], img_mat[1] * space_size[1], img_mat[2] * space_size[2]), dtype=np.float32)
    for i in range(0, img_mat[0] - step[0] + 1):
        for j in range(0, img_mat[1] - step[1] + 1):
            for k in range(0, img_mat[2] - step[2] + 1):
                img_temp = img_volume[i * space_size[0]: i * space_size[0] + img_size[0], j * space_size[1]: j * space_size[1] + img_size[1], k * space_size[2]: k * space_size[2] + img_size[2], :]
                # img_temp1 = np.repeat(img_temp, 5, axis=0)
                # raw_img = sitk.GetImageFromArray(img_temp)
                # sitk.WriteImage(raw_img, 'test1.tif')
                img = ((torch.tensor(img_temp.astype(np.float32)) / 255).permute(3,0,1,2)).unsqueeze(0)
                if torch.cuda.is_available():
                    img = img.to(device)
                with torch.no_grad():
                    output = model(img)
                    result = nn.functional.softmax(output[0]).detach().cpu().numpy()
                    # if output[0,0]>output[0,1]:
                    #     label = 0
                    # else:
                    #     label = 1
                    # del output
                # img2 = (lectin_anti_norm * 65535).astype(np.uint16)
                # lectin_img = sitk.GetImageFromArray(img2)
                # sitk.WriteImage(lectin_img, 'test2.tif')
                target[i * space_size[0]: i * space_size[0] + img_size[0], j * space_size[1]: j * space_size[1] + img_size[1], k * space_size[2]: k * space_size[2] + img_size[2]] += result[1]
                adjust[i * space_size[0]: i * space_size[0] + img_size[0], j * space_size[1]: j * space_size[1] + img_size[1], k * space_size[2]: k * space_size[2] + img_size[2]] += 1
                # target.paste(fake_img, (j * img_size, i * img_size))
                # del output
                print(i, j, k)

    return target, adjust

if __name__ == "__main__":

    img_path = '20250923_1mm new/G10_HE.tif'
    net_path = 'time_20250825114838_HE twoclass 3D 50 120 120 50um z-1um/HE twoclass 3D 50 120 120 50um z-1um_epochs_20.pkl'

    model = CNN3D_1um(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])

    if torch.cuda.is_available():
        model.to(device)
        print('GPU is ok')

    img_raw = io.imread(img_path)
    img_mat = [int(img_raw.shape[0] / space_size[0]), int(img_raw.shape[1] / space_size[1]), int(img_raw.shape[2] / space_size[2])]
    print('mat_size:', img_mat)
    target, adjust = get_volume(model=model, img_mat=img_mat, img_volume=img_raw)

    mask_norm = target / adjust
    mask_anti_norm = (mask_norm * 255).astype(np.uint8)
    # img_tensor = torch.tensor(img.astype(np.float32)) / 65535
    # img_tensor[img_tensor > 1] = 1
    # img_norm = normalize(img_tensor).unsqueeze(0).unsqueeze(0)
    # if torch.cuda.is_available():
    #     img_norm = img_norm.to(device)
    # lectin_norm = model(img_norm)
    # lectin_anti_norm = output2show16bit(torch.clip(lectin_norm, -1, 1))
    # lectin_anti_norm = output2show16bit(lectin_norm)
    # lectin_anti_norm = output2show16bit(torch.clip(lectin_norm, -1, 1))
    io.imsave('20250923_1mm new/G10_HE net20 1um 10 24 24 score.tif', mask_anti_norm)
    # lectin_img = sitk.GetImageFromArray(lectin_anti_norm)
    # sitk.WriteImage(lectin_img, 'set4_prediction_net10.tif')
    # a=1






