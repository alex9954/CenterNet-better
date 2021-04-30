import torch
import torch.nn as nn
import torchvision
import numpy as np

from ..generator import CenterNetDecoder, CenterNetGT

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
        self.dot_number = cfg.MODEL.CENTERNET.DOT_NUMBER
        self.lstm_encoder = nn.LSTM(input_size=cfg.MODEL.CENTERNET.DOT_DIMENSION,
                            hidden_size=128,
                            num_layers=1)
        self.lstm_decoder = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1)
        self.projection_x = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, 1))
        self.projection_y = nn.Sequential(nn.Linear(128, 64),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(64, 1))
        self.image_size = cfg.MODEL.CENTERNET.IMAGE_SIZE
        self.down_ratio = cfg.MODEL.CENTERNET.DOWN_SCALE
        self.dot_dimension = cfg.MODEL.CENTERNET.DOT_DIMENSION

    def forward(self, x, gt_dict, scoremap):
        x = self.extraction(x)  # (B, DOT_DIMENSION, 128, 128)
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        B, D, H, W = x.size()
        # if not self.training:  # inference
        if False:
            # x = x[:, 0:gt_dict['object_count'], :, :]
            x = x.view(B, D, H, W)
            fmap = CenterNetDecoder.pseudo_nms(scoremap)
            scores, index, ys, xs = CenterNetDecoder.topk_score(fmap, K=self.dot_number)
            object_count = gt_dict['object_count']
            for i in range(object_count.size(0)-1):
                x_point = xs[i, :int(object_count[i+1]), :]
                y_point = ys[i, :int(object_count[i+1]), :]
            index = torch.cat([x_point.unsqueeze(2), y_point.unsqueeze(2)], dim=2)
            keypoints_per_image_x = []
            keypoints_per_image_y = []
            for k in range(index.size(0)):
                dot_y = torch.stack([x[0, :, int(index[k][i][1]), int(index[k][i][0])] for i in range(index.size(1))], dim=0)
                dot_x = torch.stack([x[0, :, int(index[k][i][0]), int(index[k][i][1])] for i in range(index.size(1))], dim=0)
                keypoints_per_image_x.append(dot_x)
                keypoints_per_image_y.append(dot_y)
            keypoints_x = torch.stack(keypoints_per_image_x, dim=0)
            keypoints_y = torch.stack(keypoints_per_image_y, dim=0)

            mask_point = gt_dict['mask_point']
            mask_point = [i // self.down_ratio for i in mask_point]
            for i in range(len(mask_point)):
                padding = torch.zeros((self.dot_number - mask_point[i].size(0) // 2) * 2, device=device)
                mask_point[i] = torch.cat([mask_point[i].float().to(device), padding], dim=0)
            gt_keypoints = torch.stack(mask_point, dim=0)
            gt_keypoints = gt_keypoints.view(x.size(0), gt_keypoints.size(0), self.dot_number, 2)
        else:
            x = x.view(B, D, H * W)
            keypoints_x = []
            keypoints_y = []
            gt_keypoints = []
            object_per_image = gt_dict['object_count'].int()
            for i in range(object_per_image.size(0) - 1):
                object_per_image[i+1][0] += object_per_image[i][0]
            mask_point = gt_dict['mask_point']
            mask_point = [i // self.down_ratio for i in mask_point]

            # fetch corresponding pixels based on keypoint coordinates
            for i in range(B):
                keypoints_per_image_x = []
                keypoints_per_image_y = []
                gt_keypoints_per_image = []
                for j in range(object_per_image[i][0].item(), object_per_image[i+1][0].item()):
                    index = mask_point[j].view(-1, 2)
                    index = index.clamp(max=127.0)
                    gt_keypoints_per_image.append(index * self.down_ratio)
                    # index_1 = index[:, 1] * (self.image_size / self.down_ratio) + index[:, 0]
                    # dot_1 = torch.stack([x[i, :, int(index_1[k].item())] for k in range(index_1.size(0))], dim=0)
                    x = x.view(B, D, H, W)

                    dot_y = torch.stack([x[i, :, int(index[k][1]), int(index[k][0])] for k in range(index.size(0))], dim=0)
                    dot_x = torch.stack([x[i, :, int(index[k][0]), int(index[k][1])] for k in range(index.size(0))],
                                        dim=0)
                    keypoints_per_image_x.append(dot_x)
                    keypoints_per_image_y.append(dot_y)
                keypoints_x.append(keypoints_per_image_x)
                keypoints_y.append(keypoints_per_image_y)
                gt_keypoints.append(gt_keypoints_per_image)

            keypoints_x = self.padding(keypoints_x, device, gt=False)  # (B, max_instance, dot_number, D)
            keypoints_y = self.padding(keypoints_y, device, gt=False)
            gt_keypoints = self.padding(gt_keypoints, device, gt=True)  # (B, max_instance, dot_number, D)
        # import pdb;
        # pdb.set_trace()
        keypoints = keypoints_x.view(-1, self.dot_number, self.dot_dimension)
        keypoints = keypoints.permute(1, 0, 2)  # (dot_number, B, D)
        output_encoder, hidden_encoder = self.lstm_encoder(keypoints)
        keypoints, hidden_decoder_ = self.lstm_decoder(output_encoder, hidden_encoder)
        keypoints = keypoints.permute(1, 0, 2)  # (B, dot_number, 128)
        keypoints_xx = self.projection_x(keypoints)
        keypoints_xy = self.projection_y(keypoints)

        keypoints = keypoints_y.view(-1, self.dot_number, self.dot_dimension)
        keypoints = keypoints.permute(1, 0, 2)  # (dot_number, B, D)
        output_encoder, hidden_encoder = self.lstm_encoder(keypoints)
        keypoints, hidden_decoder_ = self.lstm_decoder(output_encoder, hidden_encoder)
        keypoints = keypoints.permute(1, 0, 2)  # (B, dot_number, 128)
        keypoints_yx = self.projection_x(keypoints)
        keypoints_yy = self.projection_y(keypoints)

        keypoints = torch.cat([keypoints_xy, keypoints_yy], dim=2)
        keypoints = keypoints.view(B, -1, self.dot_number, 2)

        return keypoints, gt_keypoints

    def padding(self, point_list, device, gt=True):
        # pad instances with less than 10 points
        for i in point_list:
            for j in range(len(i)):
                if not gt:
                    padding = torch.zeros(self.dot_number - i[j].size(0), self.dot_dimension, device=device)
                else:
                    padding = torch.zeros(self.dot_number - i[j].size(0), 2, device=device)
                i[j] = torch.cat([i[j].float().to(device), padding], dim=0)

        # pad instances per image to same size
        max_instances = torch.Tensor([len(i) for i in point_list]).max().item()
        for i in point_list:
            if not gt:
                padding = torch.zeros(int(max_instances) - len(i), self.dot_number, self.dot_dimension, device=device)
            else:
                padding = torch.zeros(int(max_instances) - len(i), self.dot_number, 2, device=device)
            i.extend(padding)
        padded_point_list = [torch.stack(i, dim=0) for i in point_list]
        padded_point_list = torch.stack(padded_point_list, dim=0)  # (B, max_instance, N, D)

        return padded_point_list


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



