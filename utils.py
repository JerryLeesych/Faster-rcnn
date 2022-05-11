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

def normalize_input(image):
    image /= 255.0
    return image

def resize_image(image, goal_size):
    w, h = goal_size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

# if we want to resized image' shorter edge to have a size of img_min_side, calc the resized image size
def get_new_img_size(height, width, img_min_side = 600):
    if width <= height:
        ratio = float(img_min_side) / width
        resized_height = int( ratio * height)
        resize_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width

class DecodeBox():
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1 # plus the background

    # convert the bboxes from ratios to pixel corrds
    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = nox_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis = -1)
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, mns_iou = 0.3, confidence_thres = 0.5 ):
        results = []
        batch_size = len(roi_cls_locs)
        rois = rois.view((batch_size, -1, 4))

        for i in range(batch_size):
            roi_cls_loc = roi_cls_locs[i] * self.std
            roi_cls_loc = roi_cls_lo.view([-1, self.num_classes, 4])

            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_box = cls_box.view([-1, (self.num_classes), 4])

            # normlize the predicted bboxes
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_scores = roi_scores[i]
            prob = F.softmax(roi_score, dim = -1)

            results.append([])

            for c in range(1, self.num_classes):
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence_thres

                # get hight prob boxes and their probs
                if c_confs[c_confs_m] > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(boxes_to_process, confs_to_process, nms_iou)

                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda  \
                                    else (c - 1) * torch.ones((len(keep), 1))

                    c_pred = torch.cat((good_boxes, confs, labels), dim = 1).cpu().numpy()
                    # similar to results += c_pred
                    results[-1].extend(c_pred)

            if results[-1]:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = results[-1][:, 0:2] + results[-1][:, 2:4])/2, \
                                reults[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results

                
