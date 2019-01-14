# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1]*(x.dim() -1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() -2)
            y = self.gamma.view(*shape)*y + self.beta.view(*shape)
        return y


# CRN : cascade refinement network
class CRN(nn.Module):
    def __init__(self, dim):
        super(CRN, self).__init__()
        ############## Layer1 ##############
        self.conv1 = nn.Conv2d(20, dim[1], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv1.weight, gain=1)
        nn.init.constant(self.conv1.bias, 0)

        self.lay1 = LayerNorm(dim[1], eps=1e-12, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv11 = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv11.weight, gain=1)
        nn.init.constant(self.conv11.bias, 0)
        self.lay11 = LayerNorm(dim[1], eps=1e-12, affine=True)
        self.relu11 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer2 ##############
        self.conv2 = nn.Conv2d(dim[1]+20, dim[2], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv2.weight, gain=1)
        nn.init.constant(self.conv2.bias, 0)

        self.lay2 = LayerNorm(dim[2], eps=1e-12, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv22 = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv22.weight, gain=1)
        nn.init.constant(self.conv22.bias, 0)
        self.lay22 = LayerNorm(dim[2], eps=1e-12, affine=True)
        self.relu22 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer3 ##############
        self.conv3 = nn.Conv2d(dim[2]+20, dim[3], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv3.weight, gain=1)
        nn.init.constant(self.conv3.bias, 0)

        self.lay3 = LayerNorm(dim[3], eps=1e-12, affine=True)
        self.relu3 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv33 = nn.Conv2d(dim[3], dim[3], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv33.weight, gain=1)
        nn.init.constant(self.conv33.bias, 0)
        self.lay33 = LayerNorm(dim[3], eps=1e-12, affine=True)
        self.relu33 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer4 ##############
        self.conv4 = nn.Conv2d(dim[3]+20, dim[4], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv4.weight, gain=1)
        nn.init.constant(self.conv4.bias, 0)

        self.lay4 = LayerNorm(dim[4], eps=1e-12, affine=True)
        self.relu4 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv44 = nn.Conv2d(dim[4], dim[4], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv44.weight, gain=1)
        nn.init.constant(self.conv44.bias, 0)
        self.lay44 = LayerNorm(dim[4], eps=1e-12, affine=True)
        self.relu44 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer5 ##############
        self.conv5 = nn.Conv2d(dim[4]+20, dim[5], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv5.weight, gain=1)
        nn.init.constant(self.conv5.bias, 0)

        self.lay5 = LayerNorm(dim[5], eps=1e-12, affine=True)
        self.relu5 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv55 = nn.Conv2d(dim[5], dim[5], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv55.weight, gain=1)
        nn.init.constant(self.conv55.bias, 0)
        self.lay55 = LayerNorm(dim[5], eps=1e-12, affine=True)
        self.relu55 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer6 ##############
        self.conv6 = nn.Conv2d(dim[5]+20, dim[6], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv6.weight, gain=1)
        nn.init.constant(self.conv6.bias, 0)

        self.lay6 = LayerNorm(dim[6], eps=1e-12, affine=True)
        self.relu6 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv66 = nn.Conv2d(dim[6], dim[6], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv66.weight, gain=1)
        nn.init.constant(self.conv66.bias, 0)
        self.lay66 = LayerNorm(dim[6], eps=1e-12, affine=True)
        self.relu66 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer7 ##############
        self.conv7 = nn.Conv2d(dim[6]+20, dim[6], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv7.weight, gain=1)
        nn.init.constant(self.conv7.bias, 0)

        self.lay7 = LayerNorm(dim[6], eps=1e-12, affine=True)
        self.relu7 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        self.conv77 = nn.Conv2d(dim[6], dim[6], kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform(self.conv77.weight, gain=1)
        nn.init.constant(self.conv77.bias, 0)
        self.lay77 = LayerNorm(dim[6], eps=1e-12, affine=True)
        self.relu77 = nn.LeakyReLU(negative_slope=1e-1, inplace=True)

        ############## Layer8 ##############
        self.conv8 = nn.Conv2d(dim[6], 27, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform(self.conv8.weight, gain=1)
        nn.init.constant(self.conv8.bias, 0)

    def forward(self, dim, label):
        out1 = self.conv1(dim[1])
        L1 = self.lay1(out1)
        out2 = self.relu1(L1)

        out11 = self.conv11(out2)
        L11 = self.lay11(out11)
        out22 = self.relu11(L11)

        m = nn.Upsample(size=(dim[1].size(3), dim[1].size(3)*2), mode='bilinear')
        img1 = torch.cat((m(out22), dim[2]), 1)

        ########################################################################

        out3 = self.conv2(img1)
        L2 = self.lay2(out3)
        out4 = self.relu2(L2)

        out33 = self.conv22(out4)
        L22 = self.lay22(out33)
        out44 = self.relu22(L22)

        m = nn.Upsample(size=(dim[2].size(3), dim[2].size(3)*2), mode='bilinear')
        img2 = torch.cat((m(out44), dim[3]), 1)

        ########################################################################

        out5 = self.conv3(img2)
        L3 = self.lay3(out5)
        out6 = self.relu3(L3)

        out55 = self.conv33(out6)
        L33 = self.lay33(out44)
        out66 = self.relu33(L33)

        m = nn.Upsample(size=(dim[3].size(3), dim[3].size(3)*2), mode='bilinear')
        img3 = torch.cat((m(out66), dim[4]), 1)

        ########################################################################

        out7 = self.conv4(img3)
        L4 = self.lay4(out7)
        out8 = self.relu4(L4)

        out77 = self.conv44(out8)
        L44 = self.lay44(out77)
        out88 = self.relu44(L44)

        m = nn.Upsample(size=(dim[4].size(3), dim[4].size(3)*2), mode='bilinear')
        img4 = torch.cat((m(out88), dim[5]), 1)

        ########################################################################

        out9 = self.conv5(img4)
        L5 = self.lay5(out9)
        out10 = self.relu5(L5)

        out99 = self.conv55(out10)
        L55 = self.lay55(out99)
        out110 = self.relu55(L55)

        m = nn.Upsample(size=(dim[5].size(3), dim[5].size(3)*2), mode='bilinear')
        img5 = torch.cat((m(out110), dim[6]), 1)

        ########################################################################

        out11 = self.conv6(img5)
        L6 = self.lay6(out11)
        out12 = self.relu6(L6)

        out111 = self.conv66(out12)
        L66 = self.lay66(out111)
        out112 = self.relu66(L66)

        m = nn.Upsample(size=(dim[6].size(3), dim[6].size(3)*2), mode='bilinear')
        img6 = torch.cat((m(out112), label), 1)

        ########################################################################

        out13 = self.conv7(img6)
        L7 = self.lay7(out13)
        out14 = self.relu7(L7)

        out113 = self.conv77(out14)
        L77 = self.lay77(out113)
        out114 = self.relu77(L77)

        ########################################################################

        out15 = self.conv8(out114)
        out15 = (out15+1.0) / 2.0 * 255.0

        out16, out17, out18 = torch.chunk(out15.permute(1, 0, 2, 3), 3.0) # permute:次元（列）の入れ替え（今回は1列目と0列目を入れ替え）
        out = torch.cat((out16, out17, out18), 1)                         # chunk:catの逆，3つに切り分ける

        return out

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        ########################################################################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max1 = nn.AvgPool2d(kernel_size=2, stride=2)
        ########################################################################
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d(kernel_size=2, stride=2)

        ########################################################################
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.max3 = nn.AvgPool2d(kernel_size=2, stride=2)

        ########################################################################
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.max4 = nn.AvgPool2d(kernel_size=2, stride=2)

        ########################################################################
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu15 = nn.ReLU(inplace=True)

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu16 = nn.ReLU(inplace=True)
        self.max5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        out1= self.conv1(x)
        out2= self.relu1(out1)
        out3= self.conv2(out2)
        out4=self.relu2(out3) # 出力
        out5=self.max1(out4)

        out6=self.conv3(out5)
        out7=self.relu3(out6) # 出力
        out8=self.conv4(out7)
        out9=self.relu4(out8) # 出力
        out10=self.max2(out9)

        out11=self.conv5(out10)
        out12=self.relu5(out11)
        out13=self.conv6(out12)
        out14=self.relu6(out13) # 出力
        out15=self.conv7(out14)
        out16=self.relu7(out15)
        out17=self.conv8(out16)
        out18=self.relu8(out17)
        out19=self.max3(out18)

        out20=self.conv9(out19)
        out21=self.relu9(out20)
        out22=self.conv10(out21)
        out23=self.relu10(out22) # 出力
        out23=self.relu10(out22)
        out24=self.conv11(out23)
        out25=self.relu11(out24)
        out26=self.conv12(out25)
        out27=self.relu12(out26)
        out28=self.max4(out27)

        out29=self.conv13(out28)
        out30=self.relu13(out29)
        out31=self.conv14(out30)
        out32=self.relu14(out31) # 出力
        out33=self.conv15(out32)
        out34=self.relu15(out33)
        out35=self.conv16(out34)
        out36=self.relu16(out35)
        out37=self.max5(out36)

        return out4, out9, out14, out23, out32, out7

if __name__ == '__main__':

    """
    Testing
    """

    vgg = VGG19()
    # model = CRN()

    x = torch.FloatTensor( np.random.random((1, 3, 256, 256)))
    # out = model(x)
    out = vgg(x)
    loss = torch.sum(out)
    loss.backward()
