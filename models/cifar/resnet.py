from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from models.custom import *

__all__ = ['resnet', 'set_gl_variable']

Linear_Class = nn.Linear
Con2d_Class = nn.Conv2d

def set_gl_variable(linear=nn.Linear, conv=nn.Conv2d):
    global Linear_Class
    Linear_Class = linear
    global Con2d_Class
    Con2d_Class = conv

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Con2d_Class(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class HW_connection(nn.Module):
    def __init__(self, planes, trans_gate_bias=0, carry_gate_bias=0, normal=False, skip_sum_1=False):
        super(HW_connection, self).__init__()
        self.normal = normal
        self.skip_sum_1 = skip_sum_1
        self.nonlinear = nn.Sigmoid()

        self.transform_gate = nn.Sequential(Con2d_Class(planes, planes, kernel_size=1, stride=1, padding=0),
                                            self.nonlinear)
        self.transform_gate[0].bias.data.fill_(trans_gate_bias)
        if not self.skip_sum_1:
            self.carry_gate = nn.Sequential(
                Con2d_Class(planes, planes, kernel_size=1, stride=1, padding=0),
                self.nonlinear)
            self.carry_gate[0].bias.data.fill_(carry_gate_bias)

    def forward(self, input_1, input_2):
        # both inputs' size maybe batch*planes*H*W
        trans_gate = self.transform_gate(input_1)  # batch*planes*H*W
        if self.skip_sum_1:
            carry_gate = 1 - trans_gate
        else:
            carry_gate = self.carry_gate(input_1)  # batch*planes*H*W

        if self.normal == True:
            l2 = torch.stack([trans_gate, carry_gate], dim=4).norm(p=2, dim=4, keepdim=False)
            trans_gate = trans_gate / l2
            carry_gate = carry_gate / l2
        output = input_2 * trans_gate + input_1 * carry_gate  # batch*opt.rnn_size

        trans_gate = trans_gate.mean()
        carry_gate = carry_gate.mean()

        return output, trans_gate, carry_gate


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, opt, skip, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.skip = skip
        if self.skip == 'HW':
            self.HW_connection = HW_connection(planes=planes*self.expansion, trans_gate_bias=opt.skip_HW_trans_bias,
                                               carry_gate_bias=opt.skip_HW_carry_bias, normal=False, skip_sum_1=opt.skip_sum_1)
        elif self.skip == 'HW-normal':
            self.HW_connection = HW_connection(planes=planes*self.expansion, trans_gate_bias=opt.skip_HW_trans_bias,
                                               carry_gate_bias=opt.skip_HW_carry_bias, normal=True, skip_sum_1=opt.skip_sum_1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.skip == 'RES':
            out += residual
        elif self.skip is not None and self.skip in ['HW', 'HW-normal']:
            out, trans_gate, carry_gate = self.HW_connection(residual, out)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, opt, skip, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Con2d_Class(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Con2d_Class(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Con2d_Class(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.skip = skip
        if self.skip == 'HW':
            self.HW_connection = HW_connection(planes=planes*self.expansion, trans_gate_bias=opt.skip_HW_trans_bias,
                                               carry_gate_bias=opt.skip_HW_carry_bias, normal=False, skip_sum_1=opt.skip_sum_1)
        elif self.skip == 'HW-normal':
            self.HW_connection = HW_connection(planes=planes*self.expansion, trans_gate_bias=opt.skip_HW_trans_bias,
                                               carry_gate_bias=opt.skip_HW_carry_bias, normal=True, skip_sum_1=opt.skip_sum_1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.skip == 'RES':
            out += residual
        elif self.skip is not None and self.skip in ['HW', 'HW-normal']:
            out, trans_gate, carry_gate = self.HW_connection(residual, out)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, opt, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.opt = opt
        self.inplanes = 16

        self.skip_2_num = opt.skip_2_num
        num_skip_2 = self._get_num_skip_2(self.skip_2_num, [n, n, n])

        self.conv1 = Con2d_Class(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, num_skip_2[0], stride=1)
        self.layer2 = self._make_layer(block, 32, n, num_skip_2[1], stride=2)
        self.layer3 = self._make_layer(block, 64, n, num_skip_2[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Linear_Class(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Linear_Class) or isinstance(m, Con2d_Class):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_skip_2, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Con2d_Class(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if 0 < blocks - num_skip_2:
            layers.append(block(self.inplanes, planes, self.opt, skip=self.opt.skip_connection_1, stride=stride,
                                downsample=downsample))
        else:
            layers.append(block(self.inplanes, planes, self.opt, skip=self.opt.skip_connection_2, stride=stride,
                                downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i < blocks - num_skip_2:
                layers.append(block(self.inplanes, planes, self.opt, skip=self.opt.skip_connection_1))
            else:
                layers.append(block(self.inplanes, planes, self.opt, skip=self.opt.skip_connection_2))

        return nn.Sequential(*layers)

    def _get_num_skip_2(self, skip_2_num, num_blocks):

        if skip_2_num <= num_blocks[2]:
            return [0, 0, skip_2_num]
        elif skip_2_num <= num_blocks[1] + num_blocks[2]:
            return [0, skip_2_num - num_blocks[2], num_blocks[2]]
        elif skip_2_num <= num_blocks[0] + num_blocks[1] + num_blocks[2]:
            return [skip_2_num - num_blocks[2] - num_blocks[1], num_blocks[1], num_blocks[2]]
        else:
            return num_blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
