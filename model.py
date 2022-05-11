import math

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import utils
import rpn
from classifier import Resnet50RoIHead, VGG16RoIHead


class FasterRCNN(nn.Moudle):
    def __init__(self, num_classes,
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'resnet50',
                    pretained = False):

        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride

        self.feature_extractor, classifier = resnet50(pretrained)

        self.rpn = RegionProposalNetwork( 1024, 512,
                                          ratios = ratios,
                                          anchor_scales = anchor_scales,
                                          feat_stride = self.feat_stride,
                                          mode = mode)

        self.head = Resnet50RoIHead( n_class = num_class + 1,
                                      roi_size = 14,
                                      spatial_scale = 1,
                                      classifier = classifier
                                      )
    def forward(self, x, scale = 1, mode = "forward"):
        if mode == "forward":

          img_size = x.shape[2:]

          # extract features from the backbone
          base_feature = self.feature_extractor.forward(x)

          # Get proposed regions
          _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)

          # do classification on proposed regions
          roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)

          return roi_cls_locs, roi_scores, rois, roi_indices

        elif mode == "extractor":
            base_featue = self.extractor.forward(x)

            return base_feature

        elif mode == "rpn":
            base_feature, img_size = x

            rpn_locs, ron_socres, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)

            return rpn_locs, rpn_scores, rois, roi_indices, anchor

        elif mode == "predict_head":
            base_feature, rois, roi_indices, img_size = x

            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)

            return roi_cls_locs, roi_scores, rois, roi_indices

        def freeze_bn(self):
            for m in self,modules():
                if  isinstance(m, nn.BatchNorm2d):
                    m.eval()


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias =False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLu(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual_in = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: # if we want to do downsample
                residual_in = self.downsample(x)

        out += residual_in
        out = self.relu(out)

class Resnet(nn.Module):
    def __init__(self, block, num_layers, num_classes = 1000):
        self.in_channels = 64
        super(Resnet, self).__init__()

        # 600, 600, 3 -> 300, 300, 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLu(inplace = True)

        # 300, 300, 64 -> 150, 150, 64
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True) # ceil the output shape
        # 150, 150, 64 -> 150, 150, 256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150, 150, 256 -> 75, 75, 512
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        # 75,75, 512 -> 38, 38, 1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        # self.layer4 will be used in classifier
        self.layer4 = self._make_layer(block, 512, layers[3], srtide = 2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if ininstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        def _make_layer(self, block, out_channels, num_layers, stride = 1):
            downsample = None

            if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels * block.expansion

            for i in range(1, num_layers):
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.vieww(x.size(0), -1)
            x = self.fc(x)

            return x

def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn. Sequential(*classifier)

    return features, classifier
