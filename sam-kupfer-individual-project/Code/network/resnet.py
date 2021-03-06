import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding of 1
    # With stride=1, this layer outputs the same dimensions as the input
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class NewBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(NewBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


n_kerns_multiplier = 32

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_classes=27):
        self.inplanes = n_kerns_multiplier
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            n_kerns_multiplier,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(n_kerns_multiplier)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, n_kerns_multiplier)
        self.layer2 = self._make_layer(
            block, 2 * n_kerns_multiplier, stride=2)
        self.layer3 = self._make_layer(
            block, 4 * n_kerns_multiplier, stride=2)
        self.layer4 = self._make_layer(
            block, 8 * n_kerns_multiplier, stride=2)
        self.avgpool = nn.AvgPool3d(
            (2, 3, 5), stride=1)
        self.fc = nn.Linear(8 * n_kerns_multiplier * 8, num_classes)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
