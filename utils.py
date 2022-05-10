import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

# function to convert the output of the network to bbox
def loc2bbox(anchor, loc):

    if anchor.size()[0] == 0:
        return torch.zeros((0, 4), dtype = loc.dtype)

    anchor_width = torch.unsqueeze(anchor[:, 2] - anchor[:, 0], -1)
    anchor_height = torch.unsqueeze(anchor[:, 3] - anchor[:, 1], -1)
    anchor_centerx = torch.unsqueeze(anchor[:, 0], -1) + 0.5 * anchor_width
    anchor_centery = torch.unsqueeze(anchor[:, 1], -1) + 0.5 * anchor_height

    dx = loc[:, 0 :: 4] # start from 0 take every 4 elements, so 0, 5, 10 etc
    dy = loc[:, 1 :: 4]
    dw = loc[:, 2 :: 4]
    dh = loc[:, 3 :: 4]

    center_x = dx * anchor_width + anchor_centerx
    center_y = dy * anchor_height + anchor_centery
    w = torch.exp(dw) * anchor_width
    h = torch.exp(dh) * anchor_height

    dst_bbox = torch.zeros_like(loc)
    # dst_bbox is upper-left corner and lower right corner, this format is required by torchvision.nms
    dst_bbox[:, 0::4] = center_x - 0.5 * w
    dst_bbox[:, 1::4] = center_y - 0.5 * h
    dst_bbox[:, 2::4] = center_x + 0.5 * w
    dst_bbox[:, 3::4] = center_y + 0.5 * w

    return dst_bbox

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):

    # each x, y pair is a center of a grid
    grid_x = np.arrange(0, width * feat_stride, feat_stride)
    grid_y = np.arrange(0, height * feat_stride, feat_stride)
    grid_x, grid_y = np.meshgrid(shift_x, shift_y)
    # ravel: flattern the matrix
    grid_xy = np.stack(grid_x.ravel(), grid_y.ravel(), grid_x.ravel(), shift_y.ravel())

    A = anchor_base.shape[0]
    K = grid_xy.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))

    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor



def normal_init(w, mean, stddev):
    w.weight.data.normal_(mean, stddev)
    w.bias,data.zero_()

def get_classes(classe_path):
    with open(classes_path, encoding = 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
