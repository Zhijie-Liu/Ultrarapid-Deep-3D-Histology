from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
# from model_virtual import Generator, Discriminator
from networks import ResnetGenerator, PixelDiscriminator
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse
import double_transforms


gpu = 0


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
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=10, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

img_size = 512


transform = double_transforms.Compose([
    # double_transforms.Resize(INPUT.IMG_SIZE),
    double_transforms.RandomHorizontalFlip(flip_prob=0.5),
    # double_transforms.Pad(INPUT.PADDING, 0, 0),
    # double_transforms.RandomCrop((img_size, img_size)),
    double_transforms.ToTensor(),
    # Normalize(mean=INPUT.PIXEL_MEAN, std=INPUT.PIXEL_STD)
])


# transform = transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),
#     transforms.RandomCrop(img_size, padding=0),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=(0.5, 0.5), std=(0.5, 0.5))
# ])

transform_to_image = transforms.Compose([
    transforms.Normalize(mean=(-1), std=(2)),
    transforms.ToPILImage()
    # transforms.Resize((256, 256))
])


if __name__ == "__main__":
    # base_name = 'hela'
    lipid_path = 'lipid tiles all'
    protein_path = 'protein tiles all'

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
    train_dataset = Subset(dataset, list(range(0, 6000)))
    valid_dataset = Subset(dataset, list(range(6000, 6256)))
    print('train_dataset length is', train_dataset.__len__())
    print('valid_dataset length is', valid_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    print("train_batch is", len(train_loader))
    print("test_batch is", len(valid_loader))

    # model1 = UNet(n_channels=1, n_classes=1)
    # model2 = UNet(n_channels=1, n_classes=1)
    # params = list(model1.parameters())
    # params = list(filter(lambda p: p.requires_grad, params))
    # nparams = sum([np.prod(p.size()) for p in params])
    # print('total nubmer of trainable parameters:', nparams)
    #
    # lr = 0.001
    # optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # opt = 'RMSprop'

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # if model.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    # Models
    D = PixelDiscriminator(1, params.ndf)
    G = ResnetGenerator(1, 1, params.ngf)

    # D.normal_weight_init(mean=0.0, std=0.02)
    # G.normal_weight_init(mean=0.0, std=0.02)


    # Set the logger
    # Loss function
    MSE_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    # Optimizers
    D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))


    # Training GAN
    save_initial_epoch = 8

    epoch_str_valid = 'Epoch {}. Train Loss D: {:.6f}, Train Loss G: {:.6f}, Valid Loss D: {:.6f}, Valid Loss G: {:.6f}'
    epoch_str_notvalid = 'Epoch {}. Train Loss D: {:.6f}, Train Loss G: {:.6f}'
    epoch_str_train = 'Epoch {}. Train Loss D: {:.6f}, Train Loss G: {:.6f}, step {}/{}'
    epoch_str_trained = 'Epoch {}. Train Loss D: {:.6f}, Train Loss G: {:.6f}, Valid Loss D: {:.6f}, Valid Loss G: {:.6f}, step {}/{}'
    time_str_pre = ' Time {:02d}:{:02d}:{:02d}'

    time_all = core_lzj.get_time()
    model_name = 'lipid2protein networks size512'
    tdloss, tgloss = [], []
    vdloss, vgloss = [], []
    save_path = './time_' + time_all + '_' + model_name + '/'
    file = save_path + model_name + '_epochs_' + str(params.num_epochs) + '.dat'
    core_lzj.check_folder_existence(save_path)
    f = open(file, 'w')

    device, init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        G.to(device)
        D.to(device)
        print('GPU is ok')

    print('start to train the model:')

    for epoch in range(params.num_epochs):
        D_loss_train = 0
        G_loss_train = 0
        D_loss_valid = 0
        G_loss_valid = 0
        # train_loss1, train_loss2 = 0, 0
        #
        # valid_loss1, valid_loss2 = 0, 0
        train_step = 0
        valid_step = 0
        prev_time = datetime.now()
        D.train()
        G.train()
        for input, target in train_loader: # fs lipid protein
            if torch.cuda.is_available():
                input = input.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                # output2 = output2.to(device, dtype=torch.float32)
            # output_true = torch.cat([output1, output2], dim=1)
            D_real = D(target).squeeze()
            real_musk = torch.ones(D_real.size()).to(device, dtype=torch.float32)
            D_real_loss = MSE_loss(D_real, real_musk)


            G_fake = G(input)
            D_fake = D(G_fake).squeeze()
            fake_musk = torch.zeros(D_fake.size()).to(device, dtype=torch.float32)
            D_fake_loss = MSE_loss(D_fake, fake_musk)

            D_loss = (D_real_loss + D_fake_loss) * 0.5
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            G_fake = G(input)
            D_fake = D(G_fake).squeeze()
            G_fake_loss = MSE_loss(D_fake, real_musk)
            l1_loss = params.lamb * L1_loss(G_fake, target)

            G_loss = G_fake_loss + l1_loss
            # G_loss = G_fake_loss
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            D_loss_train += D_loss.item()
            G_loss_train += G_loss.item()




            # output_net1,  output_net2= model1(input1), model2(input1)
            # loss1, loss2 = criterion(output_net1, output1), criterion(output_net2, output2)
            # train_loss1 += loss1.item()
            # train_loss2 += loss2.item()
            #
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # loss1.backward()
            # loss2.backward()
            # optimizer1.step()
            # optimizer2.step()

            train_step += 1
            epoch_str = epoch_str_train.format(epoch + 1,
                                               D_loss_train / train_step,
                                               G_loss_train / train_step,
                                               train_step, len(train_loader))
            print(epoch_str, end='\r')

        tdloss.append(D_loss_train / len(train_loader))
        tgloss.append(G_loss_train / len(train_loader))

        # if (epoch + 1) == int(save_initial_epoch):
        #     save_initial_epoch *= 1.2
        if (epoch + 1) % 5 == 0:
            name = model_name + '_epochs_' + str(epoch + 1) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1,
                'D': D.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'G': G.state_dict(),
                'G_optimizer': G_optimizer.state_dict()
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        if valid_loader is not None:
            D.eval()
            G.eval()
            with torch.no_grad():
                for input, target in valid_loader:  # fs lipid protein
                    if torch.cuda.is_available():
                        input = input.to(device, dtype=torch.float32)
                        target = target.to(device, dtype=torch.float32)

                # for input1, output1, output2 in valid_loader:
                #     if torch.cuda.is_available():
                #         input1 = input1.to(device, dtype=torch.float32)
                #         output1 = output1.to(device, dtype=torch.float32)
                #         output2 = output2.to(device, dtype=torch.float32)
                #     output_net1, output_net2 = model1(input1), model2(input1)
                #     loss1, loss2 = criterion(output_net1, output1), criterion(output_net2, output2)
                    D_real = D(target).squeeze()
                    real_musk = torch.ones(D_real.size()).to(device, dtype=torch.float32)
                    D_real_loss = MSE_loss(D_real, real_musk)

                    G_fake = G(input)
                    D_fake = D(G_fake).squeeze()
                    fake_musk = torch.zeros(D_fake.size()).to(device, dtype=torch.float32)
                    D_fake_loss = MSE_loss(D_fake, fake_musk)

                    D_loss = (D_real_loss + D_fake_loss) * 0.5
                    # D.zero_grad()
                    # D_loss.backward()
                    # D_optimizer.step()

                    G_fake_loss = MSE_loss(D_fake, real_musk)
                    l1_loss = params.lamb * L1_loss(G_fake, target)

                    G_loss = G_fake_loss + l1_loss
                    # G_loss = G_fake_loss
                    # G.zero_grad()
                    # G_loss.backward()
                    # G_optimizer.step()

                    D_loss_valid += D_loss.item()
                    G_loss_valid += G_loss.item()
                    # valid_loss1 += loss1.item()
                    # valid_loss2 += loss2.item()

                    valid_step += 1
                    epoch_str = epoch_str_trained.format(epoch + 1,
                                                         D_loss_train / len(train_loader),
                                                         G_loss_train / len(train_loader),
                                                         D_loss_valid / valid_step,
                                                         G_loss_valid / valid_step,
                                                         valid_step, len(valid_loader))
                    print(epoch_str, end='\r')

            vdloss.append(D_loss_valid / len(valid_loader))
            vgloss.append(G_loss_valid / len(valid_loader))

            epoch_str = epoch_str_valid.format(epoch + 1,
                                               D_loss_train / len(train_loader),
                                               G_loss_train / len(train_loader),
                                               D_loss_valid / len(valid_loader),
                                               G_loss_valid / len(valid_loader))
        else:
            # epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
            #              (epoch, train_loss / len(train_data),
            #               train_acc / len(train_data)))
            epoch_str = epoch_str_notvalid.format(epoch + 1,
                                                  D_loss_train / len(train_loader),
                                                  G_loss_train / len(train_loader))

        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)

        plottdloss, plottgloss = np.array(tdloss), np.array(tgloss)
        plotvdloss, plotvgloss = np.array(vdloss), np.array(vgloss)

        plot_all = np.vstack((plottdloss, plottgloss, plotvdloss, plotvgloss))
        plotdata = pd.DataFrame(data=plot_all.T, columns=['train loss D', 'train loss G', 'valid loss D', 'valid loss G'])
        plotdata.to_csv(save_path + model_name + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)




