# coding: utf-8
import torch
import torch.nn as nn

def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2) # 転置
    gram = torch.bmm(feat, feat_t) / (ch * h * w) # 内積（元行列，転置行列） / 1batchあたりの値
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class PerceptailLoss(nn.Module):
    def __init__(self, extractor): # extractor = VGG16
        super(PerceptailLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, output, gt):
        loss_dict = {}

        oss_dict['content'] = self.l1(output, gt)

        if output.shape[1] == 3:
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['aaa'] = 0.0
        for i in range(5):
            loss_dict['aaa'] += self.l1(feat_output[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(5):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict

if __name__ == '__main__':
    import numpy as np
    import torch
    from networks import VGG19FeatureExtractor
    size = 256
    img = torch.FloatTensor( np.random.random((1, 3, size, size))) # (batch_size, channels, width, height)
    output = torch.FloatTensor( np.random.random((1, 3, size, size))) # (batch_size, channels, width, height)
    gt = torch.FloatTensor( np.random.random((1, 3, size, size))) # (batch_size, channels, width, height)
    criterion = InpaintingLoss(VGG16FeatureExtractor())
    loss_dict = criterion(img, output, gt)
