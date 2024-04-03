import torch
from torchvision import models
from IPython.display import Code
import inspect

from fpn import FPN
from detector import Detector

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import sys


class FPN(nn.Module):
    def __init__(self, block_expansion=1, backbone="resnet18"):
        super().__init__()
        assert hasattr(models, backbone), "Undefined encoder type"
        
        # load model 
        self.feature_extractor = getattr(models, backbone)(pretrained=True)
        
        # two more layers conv6 and conv7 on the top of layer4 (if backbone is resnet18)
        
        self.conv6 = nn.Conv2d(
            512 * block_expansion, 64 * block_expansion, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=2, padding=1)

        # lateral layers
        
        self.latlayer1 = nn.Conv2d(
            512 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            256 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            128 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )

        # top-down layers
        self.toplayer1 = nn.Conv2d(
            64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=1, padding=1
        )
        self.toplayer2 = nn.Conv2d(
            64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=1, padding=1
        )

    @staticmethod
    def _upsample_add(x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, height, width = y.size()
        return F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # bottom-up
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        layer1_output = self.feature_extractor.layer1(x)
        layer2_output = self.feature_extractor.layer2(layer1_output)
        layer3_output = self.feature_extractor.layer3(layer2_output)
        layer4_output = self.feature_extractor.layer4(layer3_output)

        output = []
        
        # conv6 output. input is output of layer4
        embedding = self.conv6(layer4_output)
        
        # conv7 output. input is relu activation of conv6 output
        output.append(self.conv7(F.relu(embedding)))
        output.append(embedding)
        
        # top-down
        output.append(self.latlayer1(layer4_output))
        output.append(self.toplayer1(self._upsample_add(output[-1], self.latlayer2(layer3_output))))
        output.append(self.toplayer2(self._upsample_add(output[-1], self.latlayer3(layer2_output))))
        
        return output[::-1]


class Detector(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=2):
        super(Detector, self).__init__()
        self.fpn = FPN()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for feature_map in fms:
            loc_pred = self.loc_head(feature_map)
            cls_pred = self.cls_head(feature_map)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, 4
            )  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, self.num_classes
            )  # [N,9*2,H,W] -> [N,H,W,9*20] -> [N,H*W*9,2]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            
        
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    @staticmethod
    def _make_head(out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(64, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


if __name__ == "__main__":


    fpn = FPN()

    image_inputs = torch.rand((2, 3, 256, 256))
    # output = fpn(image_inputs)
    # print(output)

    detector = Detector(12)

    location_pred, class_pred = detector(image_inputs)
    print(location_pred.size())
    print(class_pred.size())
    print()