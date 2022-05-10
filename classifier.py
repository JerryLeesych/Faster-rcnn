import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool
import utils

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super().__init__()
        self.classifier = classifier

        self.cls_loc = nn.linear(4096, n_class * 4)
        self.score = nn.linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # roi_size here is the output size of roipool
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()


        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim = 1)

        pool = self.roi(x, indices_and_rois)

        fc7 = self.classifier(pool)

        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_score = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_score = roi_score.view(n, -1, roi_score.size(1))

        return roi_cls_locs, roi_score

class Resnet50RoIHead(nn.module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super().__init__()
        self.classifier = classifier

        self.cls_loc = nn.linear(2048, n_class * 4)
        self.score = nn.linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # roi_size here is the output size of roipool
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()


        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim = 1)

        pool = self.roi(x, indices_and_rois)

        fc7 = self.classifier(pool)

        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_score = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_score = roi_score.view(n, -1, roi_score.size(1))

        return roi_cls_locs, roi_score
