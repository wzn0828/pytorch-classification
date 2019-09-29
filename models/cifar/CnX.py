import torch.nn as nn
import torch.nn.init as init
import models.custom as custom

__all__ = ['cnx']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return custom.Con2d_Class(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Layer(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Layer, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        return out



class CnX(nn.Module):
    def __init__(self, num_classes=1000):
        super(CnX, self).__init__()

        self.layer1 = Layer(3, 32)
        self.layer2 = Layer(32, 32)
        self.layer3 = Layer(32, 32)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = custom.Linear_Class(512, num_classes)

        for m in self.modules():
            if isinstance(m, custom.Linear_Class) or isinstance(m, custom.Con2d_Class):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, custom.BN_Class):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cnx(**kwargs):
    """
    Constructs a CnX model.
    """
    return CnX(**kwargs)