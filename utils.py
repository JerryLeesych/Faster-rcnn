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
