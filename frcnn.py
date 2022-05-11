import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import model
import utils
from utils import DecodeBox


class FRCNN(object):
    _defaults = {

        "model_path"    : 'model_data/voc_weights_resnet.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        "backbone": "resnet50",
        "confidence_thres": 0.5,
        "nms_iou": 0.3,
        "anchors_scales": [8, 16, 32],
        "cuda": True

    }

    # this is equal to set get_defaults = classmethod(get_defaults)
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # initialize the faster RCNN
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            # set the attributes of self
            setattr(self, name, value)

        # get the class and num of the prior boxes
        self.class_names, self.num_class = get_classes(self.classes_path)
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]

        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # set colors for the boxes

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):

        # Load model and weights
        self.net = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location = device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            #=======
            # nn.DataParallel
            # This container parallelizes the application of the given module by splitting the input across the specified
            # devices by chunking in the batch dimension (other objects will be copied once per device).
            # In the forward pass, the module is replicated on each device,
            # and each replica handles a portion of the input.
            #  During the backwards pass, gradients from each replica are summed into the original module.
            #  The batch size should be larger than the number of GPUs used.
            #======

            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image, crop = False):

        image_shape = np.array(np.shape(image)[0:2])

        # compute the image shape needed for the model(short edge is 600)
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # convert to RGB
        image = cvtColor(image)
        # reszie the image
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        # add the dim for batchsize
        image_data = np.expand_dims(np.transpose(normalize_input(np.array(image_data, dtype = 'float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

        # roi_cls_locs: offsets to bbox based on anchor
        # roi decoded roi_cls_locs

        roi_cls_locs, roi_scores, rois, _ = self.net(images)
        results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, \
                                            nms_iou = self.nms_iou, confidence_thres = self.confidence_thres)

        # if no object is detected, return the image
        if not results[0]:
            return image

        top_label = np.array(results[0][:, 5], dtype = 'int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font = 'model_data/simhei.ttf', \
                            size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

        # if we want to crop the detected object
        if crop:
            for i, c in list(enumerate(top_label)):
                # get the 4 point of the bbox
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image_size[1], np.floor(bottom).astype('int32'))
                right = min(image_size[0], np.floor(right).astype('int32'))

                image_save_path = "img_crop"

                if not os.path.exists(image_save_path):
                    os.makedirs(dir_save_path)

                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(image_save_path, "crop_" + str(i) + ".png"), quality = 95, subsampling = 0)
                print("save crop_" + str(i) + ".png to " + image_save_path)

        # draw bbox on the image
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image_size[1], np.floor(bottom).astype('int32'))
            right = min(image_size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline = self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill = self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill = (0, 0, 0), font = font)
            del draw

        return image

    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image = cvtColor(image)

        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(np.transpose(prerpocess_input(np.array(image_data, dtype = 'float32')), \
                                        (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, \
                                            nms_iou = self.nms_iou, confidence_thres = self.confidence_thres)

        t1 = time.time()
        for _ in range(test_intervel):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, \
                                                nms_iou = self.nms_iou, confidence_thres = self.confidence_thres)

        t2 = time.time()
        return test_interval / (t2 - t1)

    # detect the image
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), 'w')
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image = cvtColor(image)

        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(np.transpose(prerpocess_input(np.array(image_data, dtype = 'float32')), \
                                        (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, \
                                nms_iou = self.nms_iou, confidence_thres = self.confidence_thres)
            if not results[0]:
                return

            top_label = np.array(results[0][:, 5], dtype = "int32")
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i ,c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            scroe = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s, %s, %s, %s, %s, %s\n" % (predicted_class, score[:6], str(int(left)), \
                                                    str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return     
