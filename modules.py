import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
  def __init__(self, ninputs, noutputs, width=32, depth=3, 
               batchnorm=False, dropout=False):
    super(FullyConnected, self).__init__()

    if dropout:
      raise NotImplemented

    layers = []
    for d in range(depth-1):
      if d == 0:
        _in = ninputs
      else:
        _in = width
      if batchnorm:
        fc = nn.Linear(_in, width, bias=False)
        fc.bias.data.zero_()
        nn.init.xavier_uniform(fc.weight.data, nn.init.calculate_gain('relu'))
        bn = nn.BatchNorm2d(width)
        bn.bias.data.zero_()
        bn.weight.data.fill_(1.0)
        layers.append(fc)
        layers.append(bn)
      else:
        fc = nn.Linear(_in, width, bias=True)
        fc.bias.data.zero_()
        nn.init.xavier_uniform(fc.weight.data, nn.init.calculate_gain('relu'))
        layers.append(fc)
      layers.append(nn.ReLU(inplace=True))

    fc = nn.Linear(width, noutputs)
    fc.bias.data.zero_()
    nn.init.xavier_uniform(fc.weight.data)
    layers.append(fc)

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    x = self.net(x)
    return x


class LinearChain(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=32, depth=3,
               pad=True, batchnorm=False):
    super(LinearChain, self).__init__()
    if pad:
      padding = ksize//2
    else:
      padding = 0
    layers = []
    for d in range(depth-1):
      if d == 0:
        _in = ninputs
      else:
        _in = width
      # conv = nn.Conv2d(_in, width, ksize, padding=padding, bias=True)
      # conv.bias.data.zero_()
      # nn.init.xavier_uniform(conv.weight.data, nn.init.calculate_gain('relu'))
      layers.append(ConvBNRelu(_in, ksize, width, batchnorm=batchnorm))
      # layers.append(nn.ReLU(inplace=True))

    if depth > 1:
      _in = width
    else:
      _in = ninputs
    conv = nn.Conv2d(_in, noutputs, 1, bias=True)
    conv.bias.data.zero_()
    nn.init.xavier_uniform(conv.weight.data)
    layers.append(conv)

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class ConvBNRelu(nn.Module):
  def __init__(self, ninputs, ksize, noutputs, batchnorm=True, stride=1):
    super(ConvBNRelu, self).__init__()
    if batchnorm:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=ksize//2, bias=False)
      bn = nn.BatchNorm2d(noutputs)
      bn.bias.data.zero_()
      bn.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, bn, nn.ReLU(inplace=True))
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=ksize//2)
      conv.bias.data.zero_()
      self.layer = nn.Sequential(conv, nn.ReLU(inplace=True))

    nn.init.xavier_uniform(conv.weight.data, nn.init.calculate_gain('relu'))

  def forward(self, x):
    return self.layer(x)


class SkipAutoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=32, depth=3, max_width=512, 
               batchnorm=True, grow_width=False):
    super(SkipAutoencoder, self).__init__()
    ds_layers = []
    widths = []

    self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')

    # ds_layers.append(SkipAutoencoderDownsample(ninputs, ksize, w, batchnorm=False))
    widths = []
    prev_w = ninputs
    for d in range(depth):
      if grow_width:
        _out = min(width*(2**d), max_width)
      else:
        _out = width
      ds_layers.append(ConvBNRelu(prev_w, ksize, _out, batchnorm=batchnorm, stride=2))
      widths.append(_out)
      prev_w = _out

    us_layers = []
    for d in range(depth-1, -1, -1):
      prev_w = widths[d]
      if d == 0:
        # w_input = width+ninputs
        next_w = ninputs
        _out = width
      else:
        next_w = widths[d-1]
        _out = next_w
        # w_input = width*2
      _in = prev_w + next_w
      us_layers.append(ConvBNRelu(_in, ksize, _out, batchnorm=batchnorm))

    self.ds_layers = nn.ModuleList(ds_layers)
    self.us_layers = nn.ModuleList(us_layers)

    self.prediction = nn.Conv2d(width, noutputs, 1)
    self.prediction.bias.data.zero_()
    nn.init.xavier_uniform(self.prediction.weight.data)

  def forward(self, x):
    data = []
    data.append(x)
    for l in self.ds_layers:
      x = l(x)
      data.append(x)

    x = data.pop()
    for l in self.us_layers:
      prev = data.pop()
      upsampler = nn.Upsample(size=prev.shape[-2:], mode='bilinear')
      x = upsampler(x)
      x = th.cat((x, prev), 1)
      x = l(x)

    x = self.prediction(x)
    return x
