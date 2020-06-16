import torch
import torch.nn as nn
import torch.nn.functional as F
import math


Linear_Class = nn.Linear
Con2d_Class = nn.Conv2d
BN_Class = nn.BatchNorm2d
_detach = None
_normlinear = None
_normconv2d = None
_coeff = True
_scale_linear = 16.0
_scale_large = 16.0
_scale_small = 4.0

_margin_theta = 0.0
_margin_cosine = 0.0
_detach_diff = True
_m_mode = 'fix'
_margin_type = 'all'    # 'all', 'label'

_bias = True

_feature_BN = False

_only_margin_right = False


def set_gl_variable(linear=nn.Linear, conv=nn.Conv2d, bn=nn.BatchNorm2d, detach=None, normlinear=None, normconv2d=None,
                    coeff=True, scale_linear=16.0, detach_diff=False, margin_theta=0., margin_cosine=0., m_mode='fix',
                    margin_type='all', bias=True, scale_large=16.0, scale_small=4.0, feature_BN=False,
                    only_margin_right=False):
    global Linear_Class
    Linear_Class = linear

    global Con2d_Class
    Con2d_Class = conv

    global BN_Class
    BN_Class = bn

    global _detach
    if detach is not None:
        _detach = detach

    global _normlinear
    if normlinear is not None:
        _normlinear = normlinear

    global _normconv2d
    if normconv2d is not None:
        _normconv2d = normconv2d

    global _coeff
    if coeff is not None:
        _coeff = coeff

    global _scale_linear
    if scale_linear is not None:
        _scale_linear = scale_linear

    global _detach_diff
    if detach_diff is not None:
        _detach_diff = detach_diff

    global _margin_theta
    if margin_theta is not None:
        _margin_theta = margin_theta

    global _margin_cosine
    if margin_cosine is not None:
        _margin_cosine = margin_cosine

    global _m_mode
    if m_mode is not None:
        _m_mode = m_mode

    global _margin_type
    if margin_type is not None:
        _margin_type = margin_type

    global _bias
    if bias is not None:
        _bias = bias

    global _scale_large
    if scale_large is not None:
        _scale_large = scale_large

    global _scale_small
    if scale_small is not None:
        _scale_small = scale_small

    global _feature_BN
    if feature_BN is not None:
        _feature_BN = feature_BN

    global _only_margin_right
    if only_margin_right is not None:
        _only_margin_right = only_margin_right


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(Linear, self).__init__(in_features, out_features, bias)
        # self.register_buffer('eps', torch.tensor(eps))
        self.eps = eps

    def forward(self, x):
        w_len = torch.sqrt(torch.t(self.weight.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))   # batch*1

        wx_len = torch.matmul(x_len, w_len)  # batch*num_classes
        cos_theta = (torch.matmul(x, torch.t(self.weight)) / (wx_len.clamp_(min=self.eps))).clamp_(-1.0, 1.0)    # batch*num_classes
        del wx_len

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (torch.matmul(x_len, w_len).clamp_(min=self.eps)) * cos_theta
        del x_len, w_len, cos_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps
        # self.register_buffer('eps', torch.tensor(eps))
        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):
        w_len = torch.sqrt(self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t().clamp_(min=self.eps))  # 1*out_channels
        x_len = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups).clamp_(min=self.eps))                             # batch*1*H_out*W_out

        wx_len = x_len * (w_len.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out

        cos_theta = (F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups) / wx_len.clamp_(min=self.eps)).clamp_(-1.0, 1.0)                                   # batch*out_channels*H_out*W_out
        del wx_len

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp_(min=self.eps) * cos_theta
        del x_len, w_len, cos_theta

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class LinearPR(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearPR, self).__init__(in_features, out_features, bias)
        self.register_buffer('one', torch.tensor(1.0))

    def forward(self, x):
        w_len_pow2 = torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))  # 1*num_classes
        x_len_pow2 = x.pow(2).sum(dim=1, keepdim=True)  # batch*1

        wx_len_pow_2 = torch.matmul(x_len_pow2, w_len_pow2)  # batch*num_classes
        del w_len_pow2, x_len_pow2

        pro = torch.matmul(x, torch.t(self.weight))  # batch*num_classes
        dis_ = torch.sqrt(F.relu(wx_len_pow_2 - pro.pow(2), inplace=True))  # batch*num_classes
        wx_len = torch.sqrt(F.relu(wx_len_pow_2, inplace=True))
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*num_classes

        wx_len_detach = wx_len.detach()
        del wx_len
        if _coeff:
            a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        else:
            a_1 = self.one
        del dis_
        if _coeff:
            a_2 = pro.detach() / (wx_len_detach + 1e-15)
        else:
            a_2 = torch.sign(pro)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro

        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2dPR(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dPR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))
        self.register_buffer('one', torch.tensor(1.0))

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
            F.relu(wx_len_pow_2 - pro.pow(2), inplace=True))                        # batch*out_channels*H_out*W_out
        wx_len = torch.sqrt(F.relu(wx_len_pow_2, inplace=True))                     # batch*out_channels*H_out*W_out
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*out_channels*H_out*W_out

        wx_len_detach = wx_len.detach()
        del wx_len
        if _coeff:
            a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        else:
            a_1 = self.one
        del dis_
        if _coeff:
            a_2 = pro.detach() / (wx_len_detach + 1e-15)
        else:
            a_2 = torch.sign(pro)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro
        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class LinearPR_Detach(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(LinearPR_Detach, self).__init__(in_features, out_features, bias)

        self.eps = eps
        # self.register_buffer('eps', torch.tensor(eps))
        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, x):
        w_len = torch.sqrt((torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt((x.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # batch*1

        cos_theta = (torch.matmul(x, torch.t(self.weight)) / torch.matmul(x_len, w_len).clamp_(min=self.eps)).clamp_(-1.0, 1.0)    # batch*num_classes
        abs_sin_theta = torch.sqrt(1.0 - cos_theta**2)    # batch*num_classes

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
                abs_sin_theta = abs_sin_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = torch.matmul(x_len, w_len).clamp_(min=self.eps) * (abs_sin_theta.detach()*cos_theta + cos_theta.detach()*(1.0-abs_sin_theta))
        del w_len, x_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2dPR_Detach(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2dPR_Detach, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        # self.register_buffer('eps', torch.tensor(eps))
        self.eps = eps
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):
        w_len = torch.sqrt((self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t()).clamp_(min=self.eps))  # 1*out_channels
        x_len = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len = torch.sqrt((F.conv2d(x_len, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups)).clamp_(min=self.eps))                             # batch*1*H_out*W_out

        wx_len = x_len * (w_len.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out

        cos_theta = (F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups) / wx_len.clamp_(min=self.eps)).clamp_(-1.0, 1.0)                                     # batch*out_channels*H_out*W_out
        del wx_len
        abs_sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
                abs_sin_theta = abs_sin_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp_(min=self.eps) * (
                            abs_sin_theta.detach() * cos_theta + cos_theta.detach() * (1.0 - abs_sin_theta))
        del w_len, x_len, cos_theta, abs_sin_theta

        # self.a1_min = a_1.view(a_1.size(0), -1).min(dim=1)[0].mean()
        # self.a2_max = torch.abs(a_2).view(a_2.size(0), -1).max(dim=1)[0].mean()
        # del a_1, a_2

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class SelfAttention(nn.Module):

    def __init__(self, in_features, inter_features):
        super(SelfAttention, self).__init__()
        self.linear1 = nn.Linear(in_features, inter_features, bias=True)
        self.linear2 = nn.Linear(inter_features, 1, bias=False)
        self._initialize_weights()

    def forward(self, features):
        '''
        :param features: N,n,in_features
        :return:
        '''
        lay1 = F.tanh(self.linear1(features))
        coeffi = F.softmax(self.linear2(lay1), dim=1)

        return (coeffi*features).sum(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=5.0/3.0)
                if m.bias is not None:
                    m.bias.data.zero_()


class SEModule(nn.Module):

    def __init__(self, channelNum):
        super(SEModule, self).__init__()
        self.linear1 = nn.Linear(channelNum, channelNum, bias=True)
        self.linear2 = nn.Linear(channelNum, channelNum, bias=False)
        self._initialize_weights()

    def forward(self, features):
        '''
        :param features: N,n,in_features
        :return:
        '''
        lay1 = self.linear1(F.avg_pool1d(features, features.size(2)).squeeze(dim=2))
        lay1 = F.tanh(lay1)
        coeffi = F.softmax(self.linear2(lay1), dim=1)
        # coeffi = F.sigmoid(self.linear2(lay1))

        return (coeffi.unsqueeze(dim=-1) * features).sum(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=5.0/3.0)
                if m.bias is not None:
                    m.bias.data.zero_()



class LinearNorm(nn.Linear):

    def __init__(self, in_features, out_features, eps=1e-8):
        if _normlinear in ['22', '23', '24', '25', '26', '27']:
            super(LinearNorm, self).__init__(out_features, out_features, _bias)
        else:
            super(LinearNorm, self).__init__(in_features, out_features, _bias)
        # self.register_buffer('eps', torch.tensor(eps))

        self.eps = eps
        self.lens = nn.Parameter(torch.ones(out_features, 1))
        self.x = []
        self.g = nn.Parameter(torch.ones(out_features, 1))
        self.out_features = out_features

        if _normlinear == '17':
            self.weight.register_hook(lambda grad: self.lens * grad)

        if _normlinear == '23':
            self.scale = nn.Parameter(_scale_linear*torch.ones(1, 1))

        if _normlinear == '26':
            self.attention = SelfAttention(out_features, 10)

        if _normlinear == '27':
            self.attention = SEModule(in_features // out_features)

        elif _normlinear == '9':
            self.register_buffer('v', self.weight.data.new_full((1, 1), _scale_linear))

    def forward(self, x):

        # weigth length
        lens = torch.sqrt((self.weight.pow(2).sum(dim=1, keepdim=True)).clamp(min=self.eps))  # out_feature*1
        self.lens.data = lens.data
        feature_len = x.norm(dim=1, keepdim=True)

        if _normlinear == '17':
            x_ = x
            weight = self.weight / (lens.detach())  # out_feature*512

        elif _normlinear == '9':
            x_ = self.v * x / feature_len  # batch*512
            weight = self.weight / lens  # out_feature*512
            self.g.data = lens.data

        elif _normlinear == '20':
            x_ = x
            weight = self.weight / lens  # out_feature*512

        #　fix the classifier
        elif _normlinear == '21':
            weight = self.weight.detach()
            x_ = x
            self.g.data = lens.data

        # 　fix the classifier　and use loaded weight
        elif _normlinear in ['22', '23', '24']:
            self.g.data = lens.data
            weight = self.weight.detach()
            x_ = x.unsqueeze(dim=1)
            x_ = F.adaptive_avg_pool1d(x_, self.out_features).squeeze(dim=1)

            if _normlinear == '23':
                scale = self.scale.detach()
                x_ = scale * F.normalize(x_, p=2, dim=1)

        #　ensamble
        elif _normlinear == '25':
            self.g.data = lens.data
            weight = self.weight.detach()

            x_ = x.unsqueeze(dim=1)
            x_ = F.adaptive_avg_pool1d(x_, x_.size(-1)//self.out_features*self.out_features).squeeze(dim=1)
            x_ = x_.view(x_.size(0), self.out_features, -1).mean(dim=-1)

        #　SelfAttention
        elif _normlinear in ['26', '27']:
            self.g.data = lens.data
            weight = self.weight.detach()

            x_ = x.unsqueeze(dim=1)
            x_ = F.adaptive_avg_pool1d(x_, x_.size(-1) // self.out_features * self.out_features).squeeze(dim=1)
            # x_ = x_.view(x_.size(0), self.out_features, -1).transpose(-2, -1)
            x_ = x_.view(x_.size(0), -1, self.out_features)

            x_ = self.attention(x_)

        elif _normlinear is None:
            weight = self.weight
            x_ = x
            self.g.data = lens.data

        else:
            raise AssertionError('_norm is not valid!')

        # costheta
        cos_theta = torch.mm(x_ / x_.norm(dim=1, keepdim=True), (weight/weight.norm(dim=1, keepdim=True)).t())  # B x class_num#
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability     # B x class_num

        self.x = []
        self.x.append(x_)

        return F.linear(x_, weight, self.bias), cos_theta


class Conv2dNorm(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2dNorm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps
        self.lens = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))
        if _normconv2d == '3-1' or _normconv2d is None:
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        if _normconv2d == '3-5':
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
            self.weight.register_hook(lambda grad: self.lens/self.g*grad)
        elif _normconv2d == '3-2':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '3-3':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.g.register_hook(lambda grad: grad/out_channels)
        elif _normconv2d == '3-4':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.g.register_hook(lambda grad: grad/math.sqrt(out_channels))
        elif _normconv2d == '4' or _normconv2d == '7':
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '5-1':
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '5-2':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '8':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):

        lens = torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)       # out*1*1*1
        self.lens.data = lens.data

        if _normconv2d == '1':
            weight = self.weight / lens  # out*in*H*W

        elif _normconv2d == '2':
            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '3-1' or _normconv2d == '3-2' or _normconv2d == '3-3' or _normconv2d == '3-4' or _normconv2d == '3-5':
            weight = torch.abs(self.g) * self.weight / lens  # out*in*H*W

        elif _normconv2d == '4':
            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '5-1' or _normconv2d == '5-2':
            weight = torch.abs(self.g) * self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, weight, None, self.stride, self.padding, self.dilation,
                                               self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '6':
            weight = self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '7':
            weight = self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '8':
            weight = lens.mean(dim=0, keepdim=True) * self.weight / lens
            self.g.data = lens.mean(dim=0, keepdim=True)

        elif _normconv2d is None:
            weight = self.weight
            self.g.data = torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)

        else:
            raise AssertionError('_norm is not valid!')

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



def LengthNormalization(weight, eps=1e-8):
    # r"weight:with shape [out_features, in_features] for fully connected layer,
    #  or [out_channels, in_channels, H, W] for convolutional layer
    weight_lens = weight.view(weight.size(0), -1).norm(dim=1, keepdim=True).clamp_(min=eps)

    if weight.dim() > 2:    # for convolutional layer
        weight_lens = weight_lens.unsqueeze(-1).unsqueeze(-1)

    return weight_lens.mean(dim=0, keepdim=True) * weight / weight_lens



class Linear_Norm37(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gi, a, x):
        '''
        :param ctx:
        :param gi: out_features x 1
        :param a: batch x out_features
        :param x: batch x in_features
        :return:
        '''
        ctx.save_for_backward(gi, a, x)

        return a * gi.t()     # batch x out_features


    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: batch x out_features
        :return:
        '''
        gi, a, x = ctx.saved_tensors

        x_len = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=1e-8))  # batch*1
        d_gi = (grad_output * torch.sign(a) * x_len).sum(dim=0, keepdim=True).t()   # out_features x 1

        d_a = grad_output * gi.t()

        return d_gi, d_a, None

linear_norm37 = Linear_Norm37.apply


class Linear_Norm16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, v, eps=1e-8):
        '''
        :param ctx:
        :param x: batch x dim_feature
        :param weight: num_classes x dim_feature
        :param bias: (num_classes, )
        :param v: scale
        :return:
        '''

        weight_lens = torch.sqrt((weight.pow(2).sum(dim=1, keepdim=True)).clamp(min=eps))  # num_classes*1

        feature_lens = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=eps))  # batch*1

        normalized_x = x / feature_lens  # batch*dim_feature

        normalized_weight = weight / weight_lens  # num_classes x dim_feature

        out = v * normalized_x.matmul(torch.t(normalized_weight))

        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x, weight, v, weight_lens, feature_lens, normalized_x, normalized_weight)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: batch x num_classes
        :return:
        '''
        x, weight, v, weight_lens, feature_lens, normalized_x, normalized_weight = ctx.saved_tensors

        # the gradient of bias
        d_bias = grad_output.sum(dim=0)         # num_classes,

        # the gradient of x
        rejection_x = torch.eye(x.size(1), device=torch.device('cuda')).unsqueeze(0) - torch.matmul(normalized_x.unsqueeze(2), normalized_x.unsqueeze(1))   # 128 x 512 x 512
        grad_x = torch.matmul(rejection_x, normalized_weight.t())                   # 128 x 512 x 100
        grad_x = torch.matmul(grad_x, grad_output.unsqueeze(2)).squeeze(dim=2)      # 128 x 512
        d_x = grad_x * v / feature_lens                                             # 128 x 512

        # the gradient of weight
        rejection_w = torch.eye(weight.size(1), device=torch.device('cuda')).unsqueeze(0) #- torch.matmul(normalized_weight.unsqueeze(2), normalized_weight.unsqueeze(1))     # 100 x 512 x 512
        grad_w = torch.matmul(rejection_w, normalized_x.t())                            # 100 x 512 x 128
        grad_w = torch.matmul(grad_w, grad_output.t().unsqueeze(2)).squeeze(dim=2)      # 100 x 512
        d_w = grad_w * v        #/ weight_lens                                                  # 100 x 512

        return d_x, d_w, d_bias, None, None

linear_norm16 = Linear_Norm16.apply


# # test code
# torch.manual_seed(123)
# ori_linear = nn.Linear(512, 1024)
# torch.manual_seed(123)
# cus_linear = LinearProDis(512, 1024)
# input_linear = torch.randn(20, 512)
# ori_linear_out = ori_linear(input_linear)
# cus_linear_out = cus_linear(input_linear)
# linear_diff = ori_linear_out - cus_linear_out
# print(linear_diff.min(), linear_diff.max())
# print(ori_linear_out.min(), ori_linear_out.max())
# print(cus_linear_out.min(), cus_linear_out.max())
#
# torch.manual_seed(123)
# ori_conv = nn.Conv2d(16, 33, 3, 2)
# torch.manual_seed(123)
# cus_conv = Conv2dProDis(16, 33, 3, 2)
# input_conv = torch.randn(20, 16, 50, 100)
# ori_conv_out = ori_conv(input_conv)
# cus_conv_out = cus_conv(input_conv)
# conv_diff = ori_conv_out - cus_conv_out
# print(conv_diff.min(), conv_diff.max())
# print(ori_conv_out.min(), ori_conv_out.max())
# print(cus_conv_out.min(), cus_conv_out.max())


##################################  ArcClassify #############################################################

class ArcClassify(nn.Linear):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, in_features, out_features, eps=1e-8):
        super(ArcClassify, self).__init__(in_features, out_features, _bias)
        # self.register_buffer('eps', torch.tensor(eps))

        self.eps = eps
        self.out_features = out_features

        self.lens = nn.Parameter(torch.ones(out_features, 1))
        self.x = []

        self.detach_diff = _detach_diff
        self.m_mode = _m_mode
        self.margin_theta = _margin_theta  # the margin theta value, default is 0.5
        self.margin_cosine = _margin_cosine  # the margin cosine value, default is 0.5
        self.margin_type = _margin_type

        # self.register_buffer('v', self.weight.data.new_full((1, 1), _scale_linear))
        self.v = _scale_linear
        self.g = nn.Parameter(torch.ones(out_features, 1))

        self.only_margin_right = _only_margin_right

        if _feature_BN:
            self.feature_BN = nn.BatchNorm1d(in_features, affine=False)

    def forward(self, embbedings, label):

        if _feature_BN:
            embbedings = self.feature_BN(embbedings)

        # weight norm
        weight_lens = torch.sqrt((self.weight.pow(2).sum(dim=1, keepdim=True)).clamp(min=self.eps))  # out_feature*1
        self.lens.data = weight_lens.data
        weight = self.weight / weight_lens  # out_feature*512

        # feature norm
        feature_len = torch.sqrt(embbedings.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1
        x_ = embbedings / feature_len  # batch*512

        # cosine
        cosine_ori = torch.mm(x_, weight.t())  # B x class_num#
        cosine = cosine_ori.clamp(-1, 1)  # for numerical stability     # B x class_num

        self.x = []
        self.x.append(self.v * x_)

        if not self.training or (self.margin_theta == 0 and self.margin_cosine == 0):
            output = cosine           # B x class_num
        elif self.margin_type == 'label':
            if self.margin_theta != 0:
                nB = len(embbedings)    # Batchsize

                # pick the labeled cos theta
                idx_ = torch.arange(0, nB, dtype=torch.long)
                labeled_cos = cosine[idx_, label]            # B
                # compute labeled theta
                labeled_theta = torch.acos(labeled_cos)         # B
                #compute the added margin
                m = self.get_m(labeled_theta)
                if self.only_margin_right:
                    m *= (cosine.argmax(dim=1)==label).to(torch.float)
                # add margin
                labeled_theta += m                         # B
                labeled_theta.clamp_(max=math.pi)
                # compute the diff and expand it
                labeled_diff = torch.cos(labeled_theta) - labeled_cos  # B
                diff_expand = labeled_diff.unsqueeze(dim=1) * F.one_hot(label, num_classes=self.out_features).to(dtype=torch.float)  # B x class_num
                if self.detach_diff:
                    diff_expand.detach_()
                # add diff and multiply the scale
                cosine = cosine + diff_expand                     # B x class_num
            output = cosine - self.margin_cosine * F.one_hot(label, num_classes=self.out_features).to(dtype=torch.float)
        elif self.margin_type == 'all':
            if self.margin_theta != 0:
                theta = torch.acos(cosine)
                theta += self.margin_theta
                theta.clamp_(min=0.0, max=math.pi)
                diff = torch.cos(theta) - cosine
                if self.detach_diff:
                    diff.detach_()
                cosine = cosine + diff
            output = cosine - self.margin_cosine
        else:
            raise AssertionError('Invalid margin_type!')

        output *= self.v
        if self.bias is not None:
            output = output + self.bias

        return output, cosine_ori

    def get_m(self, theta):
        if self.m_mode == 'larger':
            m = (self.margin_theta * theta / math.pi).detach()
        elif self.m_mode == 'smaller':
            m = (self.margin_theta - theta * self.margin_theta / math.pi).detach()
        elif self.m_mode == 'fix':
            m = self.margin_theta
        elif self.m_mode == 'larger_sqrt':
            m = (self.margin_theta * torch.sqrt(theta)).detach()

        return m


##################################  HeatedupClassify #############################################################


class HeatedupClassify(nn.Linear):

    def __init__(self, in_features, out_features, eps=1e-8):
        super(HeatedupClassify, self).__init__(in_features, out_features, _bias)
        # self.register_buffer('eps', torch.tensor(eps))

        self.eps = eps

        self.lens = nn.Parameter(torch.ones(out_features, 1))
        self.g = nn.Parameter(torch.ones(out_features, 1))

        self.x = []

        self.scale_large = _scale_large
        self.scale_small = _scale_small

    def forward(self, x):

        # weight length
        lens = self.weight.norm(dim=1, keepdim=True)  # out_feature*1
        self.lens.data = lens.data
        self.g.data = lens.data

        # compute costheta
        weight = F.normalize(self.weight, p=2, dim=1)  # out_feature*512
        normalized_x = F.normalize(x, p=2, dim=1)
        cos_theta = torch.mm(normalized_x, weight.t())  # B x class_num#
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability     # B x class_num

        # --- adaptive feature normalization --- #
        # identify the classified wrong and right
        if self.scale_large == self.scale_small:
            scale = self.scale_large
        # else:
        #     arg_max = cos_theta.argmax(dim=1)
        #     right = arg_max == label
        #     scale = self.weight.new_ones(label.size())
        #     scale[right] = self.scale_small
        #     scale[right==False] = self.scale_large
        #     scale.unsqueeze_(dim=1)

        x_ = scale * normalized_x  # batch*512

        self.x = []
        self.x.append(x_)

        return F.linear(x_, weight, self.bias), cos_theta


