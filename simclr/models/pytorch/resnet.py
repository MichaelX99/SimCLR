"""
Inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py but reworked for simplicity.
There is something to be said for not trying to be everything to everyone
"""

import torch
import torch.nn as nn

class BottleNeckBlock(nn.Module):
    def __init__(self, input_feature_map_size, width1, width2):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_feature_map_size, width1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width1)
        self.conv2 = nn.Conv2d(width1, width1, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width1)
        self.conv3 = nn.Conv2d(width1, width2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width2)
        self.relu = nn.ReLU(inplace=True)

        self.expansion = nn.Sequential(
            nn.Conv2d(input_feature_map_size, width2, kernel_size=1, bias=False),
            nn.BatchNorm2d(width2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.expansion(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=100, width_per_group=64, input_channels=3):
        super(ResNet, self).__init__()

        self.base_width = width_per_group
        layer1 = 64
        self.conv1 = nn.Conv2d(input_channels, layer1, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(layer1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self.make_block(input_feature_map_size=layer1, width1=64, width2=256, num_blocks=3)
        self.block2 = self.make_block(input_feature_map_size=256, width1=128, width2=512, num_blocks=4)
        self.block3 = self.make_block(input_feature_map_size=512, width1=256, width2=1024, num_blocks=6)
        self.block4 = self.make_block(input_feature_map_size=1024, width1=512, width2=2048, num_blocks=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, input_feature_map_size, width1, width2, num_blocks):
        blocks = []

        block = BottleNeckBlock(input_feature_map_size, width1, width2)
        blocks.append(block)

        for _ in range(1, num_blocks):
            block = BottleNeckBlock(width2, width1, width2)
            blocks.append(block)

        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
