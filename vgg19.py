# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=False)

        self.enc_1 = nn.Sequential(*vgg19.features[:4])
        self.enc_2 = nn.Sequential(*vgg19.features[4:9])
        self.enc_3 = nn.Sequential(*vgg19.features[9:14])
        self.enc_4 = nn.Sequential(*vgg19.features[14:23])
        self.enc_5 = nn.Sequential(*vgg19.features[23:32])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)
        # print(self.enc_4)
        # print(self.enc_5)

        # fix the encoder
        for i in range(5):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):

        results = [image]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

if __name__ == '__main__':
    vgg19 = VGG19FeatureExtractor()
    x = torch.FloatTensor( np.random.random((1, 3, 224, 224))) # (batch_size, channels, width, height)
    out = vgg19(x)
    # print(out[1].size())

    """
    out[0] conv1_2
    out[1] conv2_2
    out[2] conv3_2
    out[3] conv4_2
    out[4] conv5_2
    """
