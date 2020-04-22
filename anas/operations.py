import torch
import torch.nn as nn


att_OPS = {
  'none' : lambda C, affine: Zero(stride=1),
  'skip_connect' : lambda C, affine: Identity(),
  'relu_conv_bn_1x1' : lambda C, affine: ReLUConvBN(C, C, 1, 1, 0, affine=affine),
  'sep_conv_3x3' : lambda C, affine: SepConv(C, C, 3, 1, 1, affine=affine),
  'dil_conv_3x3' : lambda C, affine: DilConv(C, C, 3, 1, 2, 2, affine=affine),
  'avg_pool_3x3' : lambda C, affine: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, affine: nn.MaxPool2d(3, stride=1, padding=1),
  'spatial_score' : lambda C, affine: ChannelPool(1, 1),
  'channel_score' : lambda C, affine: SpatialPool(C)
}


class ReLUConvBN(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    n, c, h, w = x.size()
    h //= self.stride
    w //= self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h, w).fill_(0)
    return padding


class SpatialPool(nn.Module):
    def __init__(self, C):
        super(SpatialPool, self).__init__()
        self.aa_pool = nn.AdaptiveAvgPool2d(1)
        self.am_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(C * 2, C)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = (torch.cat((self.aa_pool(x), self.am_pool(x)), dim=1)).view(b, c * 2)
        out = (self.fc(out)).view(b, c, 1, 1)
        return out.expand_as(x)


class ChannelPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(ChannelPool, self).__init__()
        self.channel_pool = lambda x : torch.cat((torch.max(x, dim=1)[0].unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)), dim=1)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1) // 2, bias=False)

    def forward(self, x):
        out = self.channel_pool(x)
        out = self.conv(out)
        return out.expand_as(x)
