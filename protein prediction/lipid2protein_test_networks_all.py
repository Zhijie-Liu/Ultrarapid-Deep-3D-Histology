from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
from networks import ResnetGenerator, PixelDiscriminator
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import cv2
import argparse
import double_transforms
from PIL import Image


gpu = 0
# val_percent = 0.2
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
# parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
# parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
# parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=2000, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=1, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

# img_size = 256

transform = double_transforms.Compose([
    # double_transforms.Resize(INPUT.IMG_SIZE),
    # double_transforms.RandomHorizontalFlip(flip_prob=0.5),
    # double_transforms.Pad(INPUT.PADDING, 0, 0),
    # double_transforms.RandomCrop((img_size, img_size)),
    double_transforms.ToTensor(),
    # Normalize(mean=INPUT.PIXEL_MEAN, std=INPUT.PIXEL_STD)
])

def output2show8bit(img):
    img1 = img[0][0].detach().cpu().numpy()
    img2 = ((img1 + 1) / 2 * 255)
    img2[img2 < 0] = 0
    img2[img2 > 255] = 255
    img3 = img2.astype(np.uint8)
    img4 = Image.fromarray(img3)

    return img4

def output2show16bit(img):
    img1 = img[0][0].detach().cpu().numpy()
    img1[img1 < -1] = -1
    img1[img1 > 1] = 1
    img2 = ((img1 + 1) / 2 * 65535).astype(np.uint16)
    img3 = Image.fromarray(img2)

    return img3

if __name__ == "__main__":
    lipid_path = 'lipid tiles all'
    protein_path = 'protein tiles all'

    select = 0
    if select == 0:
        # check_dir = core_lzj.get_directory()
        net_path = core_lzj.get_files()
    else:
        # check_dir = 'check/0/145x_12_12'
        net_path = 'time_20201228112634_U-net-two-easy/U-net-two-easyparams_RMSprop_epochs_2734.pkl'

    dataset = core_lzj.Lipid2Protein(lipid_data_path=lipid_path, protein_data_path=protein_path, transform=transform)
    print('dataset length is', dataset.__len__())
    a = dataset[0]
    # a1 = transform_to_image(a[0])
    # a1.save('a1.tif')
    # a2 = transform_to_image(a[1])
    # a2.save('a2.tif')
    # im1, im2, im3 = dataset[1]
    # plt.imshow(im1.numpy()[0], cmap='gray')
    # plt.show()
    # train_dataset = Subset(dataset, list(range(0, 6000)))
    valid_dataset = Subset(dataset, list(range(0, 6200)))
    # print('train_dataset length is', train_dataset.__len__())
    print('valid_dataset length is', valid_dataset.__len__())
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    # print("train_batch is", len(train_loader))
    print("test_batch is", len(valid_loader))

    device, init_flag = core_lzj.cuda_init(gpu)

    save_path = 'lipid2protein networks test'
    core_lzj.check_folder_existence(save_path)
    print('check is starting')
    for net_dir in net_path:
        net_name = net_dir.split('_')[-1].split('.')[0]
        path = os.path.join(save_path, net_name.__str__())
        core_lzj.check_folder_existence(path)
        G = ResnetGenerator(1, 1, params.ngf)
        G.load_state_dict(torch.load(net_dir, map_location='cuda:' + gpu.__str__())['G'])
        if torch.cuda.is_available():
            G.to(device)
        print('current net is ' + net_name.__str__())
        G.eval()
        index = 1
        for im, protein in valid_loader:
            print(index)
            if torch.cuda.is_available():
                im = im.to(device, dtype=torch.float32)
            output = G(im)
            img_lipid = output2show16bit(im)
            img_protein = output2show16bit(protein)
            img_l2p = output2show16bit(output)
            # path = os.path.join(save_path, name[0])
            # core_lzj.check_folder_existence(path)
            # img = output[0][0].cpu().detach().numpy()

            # img1, img2 = output1[0][0].cpu().detach().numpy(), output2[0][0].cpu().detach().numpy()
            # img_uint16 = (img - img.min()) / (img.max() - img.min()) * 65535
            # img_raw = im[0][0].cpu().detach().numpy()
            # img_raw_uint16 = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min()) * 65535
            # img1_uint16, img2_uint16 = img1 / img1.max() * 65535, img2 / img1.max() * 65535
            img_lipid.save(os.path.join(path, index.__str__() + '-lipid_' + net_name.__str__() + '.tif'))
            img_protein.save(os.path.join(path, index.__str__() + '-protein_' + net_name.__str__() + '.tif'))
            img_l2p.save(os.path.join(path, index.__str__() + '-l2p_' + net_name.__str__() + '.tif'))

            # cv2.imwrite(os.path.join(save_path, index.__str__() + '-lipid_' + net_name.__str__() + '.tif'), img_lipid)
            # cv2.imwrite(os.path.join(save_path, index.__str__() + '-protein_' + net_name.__str__() + '.tif'), img_protein)
            # cv2.imwrite(os.path.join(save_path, index.__str__() + '-l2p_' + net_name.__str__() + '.tif'), img_l2p)
            index += 1
            # cv2.imwrite(os.path.join(path, name[0] + '-raw.png'), img_raw_uint16.astype(np.uint16))
            # cv2.imwrite(os.path.join(path, name[0] + '-protein.png'), img2_uint16.astype(np.uint16))

            # img1_data = pd.DataFrame(data=img)
            # img2_data = pd.DataFrame(data=img2.cpu().detach().numpy())
            # img1_data.to_csv(os.path.join(path, name[0] + 'test.csv'), header=False, index=False)
            # img2_data.to_csv('test2_two_306.csv', header=False, index=False)
            #
            # a=1


