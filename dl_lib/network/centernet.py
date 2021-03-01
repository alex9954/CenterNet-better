import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_lib.layers import ShapeSpec
from dl_lib.structures import Boxes, ImageList, Instances

from .generator import CenterNetDecoder, CenterNetGT
from .loss import modified_focal_loss, reg_l1_loss

import matplotlib.pyplot as plt


class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.upsample = cfg.build_upsample_layers(cfg)
        self.scoremap_head = cfg.build_head(cfg)
        self.lstm_head = cfg.build_lstm(cfg)
        # self.cls_head = cfg.build_cls_head(cfg)
        # self.wh_head = cfg.build_width_height_head(cfg)
        # self.reg_head = cfg.build_center_reg_head(cfg)

        # backbone_shape = self.backbone.output_shape()
        # feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(self.mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            return self.inference(images)

        gt_dict = self.get_ground_truth(batched_inputs)
        # self.inference(images, gt_dict)

        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features)
        scoremap = self.scoremap_head(up_fmap)
        keypoints, gt_keypoints = self.lstm_head(up_fmap, gt_dict, scoremap)
        pred_dict = {
            "scoremap": scoremap,
            "keypoints": keypoints,
        }

        # gt_dict = self.get_ground_truth(batched_inputs)

        return self.losses(pred_dict, gt_dict, gt_keypoints)

    def losses(self, pred_dict, gt_dict, gt_keypoints):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "wh": gt width and height of boxes,
                "reg": gt regression of box center point,
                "reg_mask": mask of regression,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "cls": predicted score map
            "reg": predcited regression
            "wh": predicted width and height of box
        }
        """
        # scoremap loss
        pred_scoremap = pred_dict['scoremap']
        cur_device = pred_scoremap.device
        for k in gt_dict:
            if k != 'mask_point':
                gt_dict[k] = gt_dict[k].to(cur_device)

        loss_map = modified_focal_loss(pred_scoremap, gt_dict['score_map'])

        gt_keypoints = gt_keypoints / self.cfg.MODEL.CENTERNET.IMAGE_SIZE
        keypoints = pred_dict['keypoints']
        loss_lstm = F.l1_loss(keypoints, gt_keypoints, reduction='mean')
        # index = gt_dict['index']
        # index = index.to(torch.long)
        # # width and height loss, better version
        # loss_wh = reg_l1_loss(pred_dict['wh'], mask, index, gt_dict['wh'])
        #
        # # regression loss
        # loss_reg = reg_l1_loss(pred_dict['reg'], mask, index, gt_dict['reg'])

        loss_map *= self.cfg.MODEL.LOSS.MAP_WEIGHT
        loss_lstm *= self.cfg.MODEL.LOSS.LSTM_WEIGHT
        # loss_cls *= self.cfg.MODEL.LOSS.CLS_WEIGHT
        # loss_wh *= self.cfg.MODEL.LOSS.WH_WEIGHT
        # loss_reg *= self.cfg.MODEL.LOSS.REG_WEIGHT
        #
        # loss = {
        #     "loss_cls": loss_cls,
        #     "loss_box_wh": loss_wh,
        #     "loss_center_reg": loss_reg,
        # }
        loss = {"loss_scoremap": loss_map,
                "loss_lstm": loss_lstm,
                }
        # print(loss)
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        return CenterNetGT.generate(self.cfg, batched_inputs, )

    @torch.no_grad()
    def inference(self, images, gt_dict):
        """
        image(tensor): ImageList in dl_lib.structures
        """
        n, c, h, w = images.tensor.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        img_info = dict(center=center_wh, size=size_wh,
                        height=new_h // down_scale,
                        width=new_w // down_scale)

        pad_value = [-x / y for x, y in zip(self.mean, self.std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.tensor.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images.tensor

        features = self.backbone(aligned_img)
        up_fmap = self.upsample(features)
        pred_dict = self.scoremap_head(up_fmap)
        results = self.decode_prediction(pred_dict, img_info, images, gt_dict)

        ori_w, ori_h = img_info['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **results)

        return [{"instances": det_instance}]

    def decode_prediction(self, pred_dict, img_info, images, gt_dict):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        fmap = pred_dict["cls"]
        # reg = pred_dict["reg"]
        # wh = pred_dict["wh"]

        # boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        # # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
        # scores = scores.reshape(-1)
        # classes = classes.reshape(-1).to(torch.int64)

        # # dets = CenterNetDecoder.decode(fmap, wh, reg)
        # boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        # boxes = Boxes(boxes)
        # return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)

        gt_scoremap = gt_dict['score_map']
        gt_scoremap = gt_scoremap.cpu().numpy()[0]
        gt_scoremap *= 10
        images = images.tensor
        images = images[0].permute(1, 2, 0)
        images = images.cpu()
        fmap = fmap.cpu()
        plt.figure()
        plt.imshow(images.numpy(), origin='upper')
        plt.figure()
        plt.subplot(421)
        plt.imshow(fmap.cpu().numpy()[0][0], origin='upper')
        plt.subplot(422)
        plt.imshow(gt_scoremap[0], origin='upper')
        plt.subplot(423)
        plt.imshow(fmap.cpu().numpy()[0][1], origin='upper')
        plt.subplot(424)
        plt.imshow(gt_scoremap[1], origin='upper')
        plt.subplot(425)
        plt.imshow(fmap.cpu().numpy()[0][2], origin='upper')
        plt.subplot(426)
        plt.imshow(gt_scoremap[2], origin='upper')
        plt.subplot(427)
        plt.imshow(fmap.cpu().numpy()[0][3], origin='upper')
        plt.subplot(428)
        plt.imshow(gt_scoremap[3], origin='upper')
        plt.show()


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
