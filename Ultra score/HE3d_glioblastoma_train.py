from datetime import datetime
from torch.utils.data import DataLoader
from unet3d_12g.unet3d import CNN3D_1um
from torchvision import transforms
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse


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
parser.add_argument('--num_epochs', type=int, default=500, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=10, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

img_size = 120
img_depth = 50


# transform = double_transforms.Compose([
#     # double_transforms.Resize(INPUT.IMG_SIZE),
#     double_transforms.RandomHorizontalFlip(flip_prob=0.5),
#     # double_transforms.Pad(INPUT.PADDING, 0, 0),
#     # double_transforms.RandomCrop((img_size, img_size)),
#     double_transforms.ToTensor(),
#     # Normalize(mean=INPUT.PIXEL_MEAN, std=INPUT.PIXEL_STD)
# ])


# transform = transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),
#     transforms.RandomCrop(img_size, padding=0),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=(0.5, 0.5), std=(0.5, 0.5))
# ])

transform_to_image = transforms.Compose([
    transforms.Normalize(mean=(-1), std=(2)),
    # transforms.Resize((256, 256))
])


if __name__ == "__main__":
    # base_name = 'hela'
    normal_path = 'train data 1um/normal'
    tumor_path = 'train data 1um/glioblastoma'
    dataset = core_lzj.HE3dtwoclassgroupDataset(path1=normal_path, path2=tumor_path)

    valid1_path = 'train data 1um/random normal'
    valid2_path = 'train data 1um/random o'

    validset = core_lzj.HE3dtwoclassTestset(path1=valid1_path, path2=valid2_path)


    print('train length is', dataset.__len__())
    print('valid length is', validset.__len__())

    a = dataset[10]
    # a1 = sitk.GetImageFromArray((transform_to_image(a).numpy()*65535).astype(np.uint16))
    # sitk.WriteImage(a1, 'test.tif')

    # train_dataset = Subset(dataset, list(range(0, 6000)))
    # valid_dataset = Subset(dataset, list(range(6000, 6256)))
    # print('train_dataset length is', train_dataset.__len__())
    # print('valid_dataset length is', valid_dataset.__len__())
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=8, shuffle=True, num_workers=8)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    print("train_batch is", len(train_loader))
    print("valid_batch is", len(valid_loader))
    # print("test_batch is", len(valid_loader))
    model = CNN3D_1um(in_channels=3, num_classes=2).to(device)

    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model params: %.2f M' % (total_params / 1024 / 1024))
    print('Trainable params: %.2f M' % (trainable_params / 1024 / 1024))
    # print(f"总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,}")

    # D.normal_weight_init(mean=0.0, std=0.02)
    # G.normal_weight_init(mean=0.0, std=0.02)


    # Set the logger
    # Loss function
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()

    criterion = nn.CrossEntropyLoss()
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    # G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))


    # Training GAN
    # save_initial_epoch = 8

    epoch_str_valid = 'Epoch {}. Train Loss: {:.6f}'
    epoch_str_notvalid = 'Epoch {}. Train Loss ave: {:.6f}, Valid Loss ave: {:.6f}, Valid Loss: {:.6f}, step {}/{}'
    epoch_str_train = 'Epoch {}. Train Loss ave: {:.6f}, Train Loss: {:.6f}, step {}/{}'
    epoch_str_trained = 'Epoch {}. Train Loss: {:.6f}, Valid Loss ave: {:.6f}'
    time_str_pre = ' Time {:02d}:{:02d}:{:02d}'

    time_all = core_lzj.get_time()
    model_name = 'HE twoclass glioblastoma 3D 50 120 120 50um z-1um'
    tloss = []
    vloss = []
    save_path = './time_' + time_all + '_' + model_name + '/'
    file = save_path + model_name + '_epochs_' + str(params.num_epochs) + '.dat'
    core_lzj.check_folder_existence(save_path)
    f = open(file, 'w')

    device, init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        model.to(device)
        print('GPU is ok')

    print('start to train the model:')

    for epoch in range(params.num_epochs):
        train_loss = 0
        train_step = 0
        valid_loss = 0
        valid_step = 0
        prev_time = datetime.now()
        model.train()
        # for input, target in train_loader_lipid, train_loader_lectin:
        for input, target in train_loader:
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            prediction = model(input)
            loss = criterion(prediction, target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1

            epoch_str = epoch_str_train.format(epoch + 1,
                                               train_loss / train_step,
                                               loss.item(),
                                               train_step, len(train_loader))
            print(epoch_str, end='\r')

        tloss.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            for input, target in valid_loader:
                if torch.cuda.is_available():
                    input = input.to(device)
                    target = target.to(device)

                prediction = model(input)
                loss = criterion(prediction, target)
                valid_loss += loss.item()
                valid_step += 1

                epoch_str = epoch_str_valid.format(epoch + 1,
                                                   train_loss / len(train_loader),
                                                   valid_loss / valid_step,
                                                   loss.item(),
                                                   valid_step, len(valid_loader))
                print(epoch_str, end='\r')
        vloss.append(valid_loss / len(valid_loader))


        # if (epoch + 1) == int(save_initial_epoch):
        #     save_initial_epoch *= 1.2
        if (epoch + 1) % 5 == 0:
            name = model_name + '_epochs_' + str(epoch + 1) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        epoch_str = epoch_str_trained.format(epoch + 1,
                                              train_loss / len(train_loader),
                                              valid_loss / len(valid_loader))

        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)

        plottloss, plotvloss = np.array(tloss), np.array(vloss)


        plot_all = np.vstack((plottloss, plotvloss))
        plotdata = pd.DataFrame(data=plot_all.T, columns=['train loss', 'valid loss'])
        plotdata.to_csv(save_path + model_name + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)




