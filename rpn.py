import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator()：

    def __init__(
        self,
        mode,
        nms_iou = 0.7,
        n_train_pre_nms = 12000,  # num of bbox before nms
        n_train_post_nms = 600, # num of bbox after nms
        n_test_pre_nms = 3000,
        n_test_post_nms = 300,
        min_bbox_size = 16
    ):

        self.mode = mode # training mode or predicting/testong mode
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        self.min_bbox_size = min_bbox_size

    def __cal__(self, loc, objectness, anchor, img_size, scale = 1.):
        if self.mode = "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms

        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)
        if loc.is_cude:
            anchor = anchor.cuda()

        # roi is the bboxes draw from the network prediction
        roi = loc2bbox(anchor, loc)

        # clamp the bboxes inside the image
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])

        min_bbox_size = self.min_bbox_size * scale

        # remove the bboxes that are too small
        bbox_to_keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_bbox_size) & \
                                        ((roi[:, 3] - roi[:, 1]) >= min_bbox_size))[0]
        roi = roi[bbox_to_keep, :]
        objectness = objectness[bbox_to_keep]

        sorted_idx = torch.argsort(objectness, descending = True)

        if n_pre_nms > 0:
            sorted_idx = sorted_idx[: n_pre_nms] # keep the heightest n_pre_nms boxes

        roi = roi[sorted_idx, :]
        objectness = objectness[sorted_idx]

        bbox_to_keep = nms(roi, objectness, self.nms_iou)
        bbox_to_keep = bbox_to_keep[: n_post_nms]
        roi = roi[bbox_to_keep]

        return roi




class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        ratios,
        ancho_scales = [8, 16, 32]
        feat_stride = 16,
        mode = "traning",

    ):

        super().__init__()

        # generate all possible achors according to scales and ratios
        self.anchor_base = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        num_anchors = self.anchor_base.shape[0]

        # Do a conv first to collect features
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # This conv is to predic if there is an object
        self.objectness = nn.Conv2d(mid_channels, num_anchors*2, 1, 1, 0)

        # offset to the bbox (based on achor)
        self.loc = nn.Conv2d(mid_channels, num_anchors*4, 1, 1, 0)

        # stride btw feature points
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(mode)

        # Initialize the rpn
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_szie, scale = 1.):

        n, _, h, w = x.shape

        # First do a conv to collect features
        x = F.relu(self.conv1(x))

        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # Predic the offset of the locs
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # predict the objectness
        rpn_objectness = self.objectness(x)
        rpn_objectness = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        rpn_softmax_scores = F.softmax(rpn_objectness, dim = -1)
        rpn_objectness_prob = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_objectness_prob = rpn_objectness_prob.view(n, -1)

        # generate anchors based on anchor base and image size & feat_stride
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_objectness[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi)))
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim = 0)
        roi_indices.append(batch_index)

        return rpn_locs, rpn_objectness, rois, roi_indices, anchor








def normal_init(w, mean, stddev):
    w.weight.data.normal_(mean, stddev)
    w.bias,data.zero_()
