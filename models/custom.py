import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear

class LinearProDis(Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearProDis, self).__init__(in_features, out_features, bias)

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, x):
        w_len_pow2 = torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))  # 1*num_classes
        x_len_pow2 = x.pow(2).sum(dim=1, keepdim=True)  # batch*1

        wx_len_pow_2 = torch.matmul(x_len_pow2, w_len_pow2)  # batch*num_classes
        del w_len_pow2, x_len_pow2

        pro = torch.matmul(x, torch.t(self.weight))  # batch*num_classes
        dis_ = torch.sqrt(F.relu(wx_len_pow_2 - pro.pow(2)))  # batch*num_classes
        wx_len = torch.sqrt(F.relu(wx_len_pow_2))
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*num_classes

        wx_len_detach = wx_len.detach()
        del wx_len
        a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        del dis_
        a_2 = pro.detach() / (wx_len_detach + 1e-15)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro

        # self.a1_min = a_1.min(dim=1)[0].mean()
        # self.a2_max = torch.abs(a_2).max(dim=1)[0].mean()

        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2dProDis(Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dProDis, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):

        w_len_pow2 = self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t()  # 1*out_channels
        x_len_pow2 = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len_pow2 = F.conv2d(x_len_pow2, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups)                             # batch*1*H_out*W_out

        wx_len_pow_2 = x_len_pow2 * (w_len_pow2.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out
        del w_len_pow2, x_len_pow2

        pro = F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups)                                    # batch*out_channels*H_out*W_out

        dis_ = torch.sqrt(
            F.relu(wx_len_pow_2 - pro.pow(2)))                        # batch*out_channels*H_out*W_out
        wx_len = torch.sqrt(F.relu(wx_len_pow_2))                     # batch*out_channels*H_out*W_out
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*out_channels*H_out*W_out

        wx_len_detach = wx_len.detach()
        del wx_len
        a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        del dis_
        a_2 = pro.detach() / (wx_len_detach + 1e-15)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro

        # self.a1_min = a_1.view(a_1.size(0), -1).min(dim=1)[0].mean()
        # self.a2_max = torch.abs(a_2).view(a_2.size(0), -1).max(dim=1)[0].mean()

        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out

