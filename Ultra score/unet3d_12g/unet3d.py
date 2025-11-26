"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
# from torchsummary import summary
import torch
import core_lzj
import time

gpu = 0
device, init_flag = core_lzj.cuda_init(gpu)

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res0 = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res0)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out


class CNN3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256]) -> None:
        super(CNN3D, self).__init__()
        self.block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(level_channels[2], num_classes)

    def forward(self, input):
        # Analysis path forward feed
        out1 = self.block1(input)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.avgpool(out3)
        out5 = out4.view(out4.size(0), -1)
        out = self.fc(out5)

        return out


class CNN3D_1um(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512, 1024]) -> None:
        super(CNN3D_1um, self).__init__()
        self.block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.block4 = Conv3DBlock(in_channels=level_channels[2], out_channels=level_channels[3])
        self.block5 = Conv3DBlock(in_channels=level_channels[3], out_channels=level_channels[4])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(level_channels[4], num_classes)

    def forward(self, input):
        # Analysis path forward feed
        out1 = self.block1(input)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.avgpool(out5)
        out7 = out6.view(out6.size(0), -1)
        out = self.fc(out7)

        return out


if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = CNN3D_1um(in_channels=3, num_classes=2).to(device)
    x = torch.randn(1, 3, 50, 120, 120).to(device)
    y = model(x)
    a = 1
    # start_time = time.time()
    # summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    # print("--- %s seconds ---" % (time.time() - start_time))