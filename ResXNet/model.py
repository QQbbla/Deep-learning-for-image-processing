import torch
import torch.nn as nn
from networkx.utils.misc import groups
import torchvision.models.resnet

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_pre_group=64, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_pre_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True, groups=1, width_pre_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_pre_group = width_pre_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_nums[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, out_channel, block_nums, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, groups=self.groups, width_pre_group=self.width_pre_group))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, block_nums):
            layers.append(block(self.in_channel, out_channel, groups=self.groups, width_pre_group=self.width_pre_group))

        return nn.Sequential(*layers)#非关键词参数

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    @staticmethod
    def resnet34(num_classes=1000, include_top=True):
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    def resnet50(num_classes=1000, include_top=True):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    def resnet101(num_classes=1000, include_top=True):
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

    def resnet50_32x4d(num_classes=1000, include_top=True):
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, groups=groups, width_pre_group=width_per_group)
    def resnet101_32x8d(num_classes=1000, include_top=True):
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes, include_top=include_top, groups=groups, width_pre_group=width_per_group)




