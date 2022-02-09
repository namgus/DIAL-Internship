import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        """
        WideResNet
        Args:
            num_classes: 분류할 class의 수
            depth: 모델의 깊이
            widen_factor: Widening factor k
            drop_rate: Dropout 확률
        """
        super(WideResNet, self).__init__()

        # 각 NetworkBlock의 입력/출력 채널 수
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # - 4: WideResNet의 conv1, bn1, relu1, fc 제외
        # % 6: NetworkBlock의 bn1, conv1, relu1, bn2, conv2, relu2
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6

        # NetworkBlock을 구성할 block
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], BasicBlock, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)

        # 출력 채널 수
        self.output_channels = channels[3]

        # Weight/bias matrix 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 정규분포에서 임의로 수 추출하여 weight 채움
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.output_channels)
        return self.fc(out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int or float, in_planes: int, out_planes: int, block: type(nn.Module), stride: int,
                 drop_rate=0.0, activate_before_residual=False):
        """
        WideResNet을 구성하는 NetworkBlock
        Args:
            nb_layers: Block 내부 레이어 수
            in_planes: 입력 채널
            out_planes: 출력 채널
            block: BasicBlock
            stride: Conv2d stride
            drop_rate: Dropout 확률
            activate_before_residual: 활성함수(ReLU)를 Residual을 더하기 전 활성화할지 (원래 Residual을 더한 후 활성화)
        """
        super(NetworkBlock, self).__init__()

        self.layer = nn.Sequential()

        for i in range(int(nb_layers)):
            self.layer.add_module('block' + str(i + 1),
                                  block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1,
                                        drop_rate, activate_before_residual))

    def forward(self, x):
        return self.layer(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        """
        NetworkBlock을 구성하는 BasicBlock
        Args:
            in_planes: 입력 채널
            out_planes: 출력 채널
            stride: Conv2d stride
            drop_rate: Dropout 확률
            activate_before_residual: 활성함수(ReLU)를 Residual을 더하기 전 활성화할지 (원래 Residual을 더한 후 활성화)
        """
        super(BasicBlock, self).__init__()

        self.drop_rate = drop_rate
        self.equal_in_out_channel = (in_planes == out_planes)
        self.activate_before_residual = activate_before_residual

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1, bias=True)

        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)

        self.convShortcut = (not self.equal_in_out_channel) and nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1),
                                                                          stride=stride,
                                                                          padding=0, bias=True) or None

    def forward(self, x):
        # out = self.bn1(x)
        # out = self.relu1(out)
        # out = self.conv1(out)
        #
        # out = self.bn2(out)
        # out = self.relu2(out)
        #
        #
        #
        # out = self.conv2(out)
        #
        #
        #
        if not self.equal_in_out_channel and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out_channel else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out_channel else self.convShortcut(x), out)
