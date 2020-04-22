import torch
import torch.nn as nn
from model import a_Cell


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, genotype, stride=1):
        super(BasicBlock, self).__init__()
        self.drop_path_prob = 0.
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.att = a_Cell(genotype, planes)
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out, self.drop_path_prob)

        out += residual
        out = self.relu(out)

        return out


class AttResNet(nn.Module):
    def __init__(self, block, genotype, _n_size, num_classes=10):
        super(AttResNet, self).__init__()
        self.inplane = 16
        self.stride = 1
        self._block = block
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=_n_size, genotype=genotype, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=_n_size, genotype=genotype, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=_n_size, genotype=genotype, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, genotype, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, genotype, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


### Define networks
def att_resnet_cifar(genotype, n_size, **kwargs):
    """Constructs a Att_ResNet model.

    """
    model = AttResNet(BasicBlock, genotype, _n_size=n_size, **kwargs)
    return model
