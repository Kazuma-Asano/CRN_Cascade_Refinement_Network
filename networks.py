# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

################################################################################
def get_output_ch(resolution):
    if resolution < 128:
        ch = 1024
    else:
        ch = 512

    if resolution == 512:
        ch = 128
    elif resolution == 1024:
        ch = 32

    return ch


class CRN(nn.Module):
    def __init__(self, super_resolution=256, groups=6):
        super(CRN, self).__init__()

        self.super_resolution = super_resolution

        resolution = [4, 8, 16, 32, 64, 128, 512, 1024]
        return_labels = [True, True, True, True, True, False, False]

        if self.super_resolution >= 512:
            return_labels.append(False)
        if self.super_resolution == 1024:
            return_labels.append(False)
        return_labels = return_labels[::-1] # 配列を反転

        self.refine_block0 = refine_block(resolution[0])
        self.refine_block1 = refine_block(resolution[1])
        self.refine_block2 = refine_block(resolution[2])
        self.refine_block3 = refine_block(resolution[3])
        self.refine_block4 = refine_block(resolution[4])
        self.refine_block5 = refine_block(resolution[5])
        self.refine_block6 = refine_block(resolution[6])

        if self.super_resolution >= 512:
            self.refine_block7 = refine_block(resolution[7])
        if self.super_resolution == 1024:
            self.refine_block8 = refine_block(resolution[8])

        last_in_ch = get_output_ch(self.super_resolution)
        self.last_conv = nn.Conv2d(last_in_ch, 3, kernel_size=1)

    def forward(self, label):

        x = self.refine_block0(label)

        x = self.refine_block1(label, x)
        x = self.refine_block2(label, x)
        x = self.refine_block3(label, x)
        x = self.refine_block4(label, x)
        x = self.refine_block5(label, x)
        x = self.refine_block6(label, x)

        # if self.super_resolution >= 512:
        #     x = self.refine_block7(label, x)
        # if self.super_resolution == 1024:
        #     x = self.refine_block8(label, x)

        conv_out = self.last_conv(x)
        print(conv_out.size())

        aaa = (conv_out + 1.0) / 2.0 * 255.0
        temp = aaa.split(3, dim=1)
        result = torch.cat(temp, dim=0)
        return result

################################################################################

class refine_block(nn.Module):
    def __init__(self, super_resolution):
        super(refine_block, self).__init__()
        self.super_resolution = super_resolution

        x_grid = torch.linspace(-1, 1, 2 * self.super_resolution).repeat(self.super_resolution, 1)
        y_grid = torch.linspace(-1, 1, self.super_resolution).view(-1, 1).repeat(1, self.super_resolution * 2)

        self.grid = torch.cat((x_grid.unsqueeze(2), y_grid.unsqueeze(2)), 2).unsqueeze_(0)

        out_ch = get_output_ch(self.super_resolution)
        if self.super_resolution == 4:
            in_ch = 3
        else:
            in_ch = get_output_ch(self.super_resolution // 2) + 3

        self.module = nn.Sequential (
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[out_ch, self.super_resolution, self.super_resolution*2]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[out_ch, self.super_resolution, self.super_resolution*2]),
            nn.LeakyReLU(0.2)
        )

    def forward(self, label, x=None):
        print('label size {}'.format(label.size()))
        grid = self.grid.repeat(label.size(0), 1, 1, 1)
        print('grid size {}'.format(grid.size()))
        # print('Label size {}'.format(label.size()))
        label_downsampled = F.grid_sample(label, grid) # downsample
        print('down Label size {}'.format(label_downsampled.size()))
        if self.super_resolution != 4:

            upsample_out = F.interpolate(x, size=(self.super_resolution, self.super_resolution*2), mode='bilinear', align_corners=True)

            x = torch.cat((upsample_out, label_downsampled), 1)
        else:
            x = label_downsampled
        print('x size {}'.format(x.size()))

        module_out = self.module(x)
        print('module_out size {}'.format(module_out.size()))
        # exit()
        print('-' * 30)
        return module_out

if __name__ == '__main__':
    model = CRN(super_resolution=480)
    label = torch.FloatTensor( np.random.random((1, 3, 480, 640))) # (batch_size, channels, width, height)
    out = model(label)
    print(out.size())
