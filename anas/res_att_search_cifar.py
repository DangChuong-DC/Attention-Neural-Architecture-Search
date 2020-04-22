import torch
import torch.nn as nn
import torch.nn.functional as F
from model_search import a_Cell
from genotypes import ATT_PRIMITIVES
from genotypes import AttGenotype


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, alpha, stride=1):
        super(BasicBlock, self).__init__()
        self.p = 0.
        self._alpha = alpha
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.att = a_Cell(planes)
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def update_p(self):
        self.att.p = self.p
        self.att.update_p()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        weights = F.softmax(self._alpha, dim=-1)
        out = self.att(out, weights)

        out += residual
        out = self.relu(out)

        return out


class AttResNet(nn.Module):
    def __init__(self, block, _n_size, gpus, num_classes=10):
        super(AttResNet, self).__init__()
        self.inplane = 16
        self.nodes = 3
        self.stride = 1
        self._criterion = nn.CrossEntropyLoss()
        self._block = block
        self._gpus = gpus
        self._no_cls = num_classes
        self.n_size = _n_size
        self._initialize_alpha()
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=_n_size, alpha=self.alphas, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=_n_size, alpha=self.alphas, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=_n_size, alpha=self.alphas, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, alpha, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, alpha, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def update_p(self, dr_p):
        for layer in (self.layer1, self.layer2, self.layer3):
            for block in layer:
                block.p = dr_p
                block.update_p()

    def _initialize_alpha(self):
        k = sum(1 for i in range(self.nodes) for n in range(2+i))
        num_ops = len(ATT_PRIMITIVES)
        self.alphas = nn.Parameter(1e-3 * torch.randn(k, num_ops).cuda(),
            requires_grad=True)
        self._arch_param = nn.ParameterList([self.alphas])

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

    def new(self):
        model_new = AttResNet(self._block, self.n_size, self._gpus, self._no_cls)
        if self._gpus > 1:
            model_new = nn.DataParallel(model_new)
            model_new.cuda()
            for x, y in zip(model_new.module.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)
        else:
            model_new.cuda()
            for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)
        return model_new

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def dict_name_parameters(self):
        name_params = {}
        for k, v in self.named_parameters():
            if not k.endswith('alphas'):
                name_params[k] = v
        return name_params

    def net_parameters(self):
        return self.dict_name_parameters().values()

    def named_net_parameters(self):
        return self.dict_name_parameters().items()

    def arch_parameters(self):
        return self._arch_param

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.nodes):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != ATT_PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != ATT_PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((ATT_PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
        genotype = AttGenotype(att=gene)
        return genotype


### Define networks
def att_resnet_cifar(n_size, no_gpus, **kwargs):
    """Constructs a Att_ResNet model.

    """
    model = AttResNet(BasicBlock, _n_size=n_size, gpus=no_gpus, **kwargs)
    return model
