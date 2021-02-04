import torch
import torch.nn as nn
import torchvision
import json


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class LSTMhead(nn.Module):
    def __init__(self, cfg):
        super(LSTMhead, self).__init__()
        self.extraction = SingleHead(
            64,
            cfg.MODEL.CENTERNET.DOT_DIMENSION,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.training = cfg.TRAIN
        self.dot_number = cfg.MODEL.CENTERNET.DOT_NUMBER
        self.lstm = nn.LSTM(input_size=cfg.MODEL.CENTERNET.DOT_DIMENSION,
                            hidden_size=2,
                            num_layers=1)

    def forward(self, x, gt_dict):
        x = self.extraction(x)  # (B, DOT_DIMENSION, 128, 128)
        B, D, H, W = x.size()
        x = x.view(B, D, H * W)
        if not self.training:  # inference
            # x = x[:, 0:gt_dict['object_count'], :, :]
            keypoints, _ = torch.topk(x, self.dot_number, dim=-1)  # (B, D, dot_number)
        else:
            mask_point = gt_dict['mask_point']
            mask_point = torch.Tensor(mask_point)
            # mask_point = mask_point.squeeze(0) # used for debugging
            index = mask_point.view(mask_point.size(0), -1, 2)
            index = index[:, :, 0] * (image_size / down_ratio) + index[:, :, 1]
            dot = torch.stack([x[i, :, int(index[i][j].item())] for i in range(B) for j in range(index.size(1))])
            keypoints = dot.view(B, D, self.dot_number)

        keypoints = keypoints.permute(2, 0, 1)  # (dot_number, B, D)
        keypoints, (hn, cn) = self.lstm(keypoints)
        keypoints = keypoints.permute(1, 0, 2)  # (B, dot_number, 2)

        return keypoints


if __name__ == '__main__':
    batch_size = 32
    image_size = 512
    down_ratio = 4
    up_fmap = torch.rand(batch_size, 64, int(image_size / down_ratio), int(image_size / down_ratio))
    cfg = dict(
        MODEL=dict(
            CENTERNET=dict(
                DOT_DIMENSION=128,
                BIAS_VALUE=-2.19,
                DOT_NUMBER=10,
            ),
        ),
        TRAIN=True,
    )
    point_sample =[[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]] * batch_size]
    gt_dict = {}
    gt_dict['mask_point'] = point_sample
    lstm = LSTMhead(cfg)
    reult = lstm(up_fmap, gt_dict)



