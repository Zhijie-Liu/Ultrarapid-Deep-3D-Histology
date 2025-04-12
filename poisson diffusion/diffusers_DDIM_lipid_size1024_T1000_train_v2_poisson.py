import copy
import torch
import core_lzj
import numpy as np
import pandas as pd
from datetime import datetime
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusion_diffusers_poisson_lzj import GaussianDiffusionTrainer, GaussianDiffusionSampler, poisson_noise_torch
from diffusers import UNet1DModel

num_epochs = 200
gpu = 0
device, init_flag = core_lzj.cuda_init(gpu)
img_size = 1024
T = 1000

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(img_size, padding=0),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_to_image = transforms.Compose([
    transforms.Normalize(mean=(-1), std=(2)),
    transforms.ToPILImage()
    # transforms.Resize((256, 256))
])


if __name__ == "__main__":
    # dataset = CIFAR10(
    #         root='./data', train=True, download=True,
    #         transform=transform)
    data_path = 'lipid tiles'
    dataset = core_lzj.DDPMFolder16bit1024ResizeDataset(path=data_path)
    # a = dataset[0]
    # t1=transforms.Normalize(mean=(-1), std=(2))
    # t2=transforms.ToPILImage()
    # b=t2(a)

    print('train data length is ', dataset.__len__())

    dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=8, drop_last=True)

    print("train batch is", len(dataloader))
    noise_scheduler = DDPMScheduler(num_train_timesteps=T)
    net_model = UNet2DModel(
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

    # ema_model = copy.deepcopy(net_model)
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4)
    # lr_schedule =

    # trainer = GaussianDiffusionTrainer(
    #     net_model, 1e-4, 0.02, T).to(device)
    # net_sampler = GaussianDiffusionSampler(
    #     net_model, 1e-4, 0.02, T, img_size,
    #     'epsilon', 'fixedlarge').to(device)
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    epoch_str_notvalid = 'Epoch {}. Train Loss: {:.6f}'
    epoch_str_train = 'Epoch {}. Train Loss: {:.6f}, Current Loss: {:.6f}, step {}/{}'
    epoch_str_trained = 'Epoch {}. Train Loss: {:.6f}, step {}/{}'
    time_str_pre = ' Time {:02d}:{:02d}:{:02d}'
    time_all = core_lzj.get_time()
    model_name = 'human brain size1024 T1000 v2 poisson'

    save_path = './time_' + time_all + '_' + model_name + '/'
    file = save_path + model_name + '_epochs_' + str(num_epochs) + '.dat'
    core_lzj.check_folder_existence(save_path)
    f = open(file, 'w')
    loss_list = []

    print('start to train the model:')
    for epoch in range(num_epochs):
        loss_train = 0
        train_step = 0
        net_model.train()
        prev_time = datetime.now()
        for img in dataloader:

            x_0 = img.to(device)
            noise = poisson_noise_torch(size=(1, 1, img_size, img_size)).to(device)
            bs = x_0.shape[0]
            timesteps = torch.randint(0, T, (bs,)).long().to(device)
            noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)
            noise_pred = net_model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     net_model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
            train_step += 1
            epoch_str = epoch_str_train.format(epoch + 1,
                                               loss_train / train_step,
                                               loss.item(),
                                               train_step, len(dataloader))
            print(epoch_str, end='\r')

        loss_list.append(loss_train / len(dataloader))
        if (epoch + 1) % 1 == 0:
            # save_initial_epoch *= 1.2
            name = model_name + '_epochs_' + str(epoch + 1) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1,
                'net_model': net_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        epoch_str = epoch_str_notvalid.format(epoch + 1,
                                              loss_train / len(dataloader))
        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)

        plot_loss_lost = np.array(loss_list)
        plot_all = np.vstack((plot_loss_lost))
        plot_data = pd.DataFrame(data=plot_all, columns=['train loss'])
        plot_data.to_csv(save_path + model_name + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)