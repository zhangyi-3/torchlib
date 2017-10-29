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
  def __init__(self, ninputs, noutputs, ksize=3, width=32, depth=3, pad=True):
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
      conv = nn.Conv2d(_in, width, ksize, padding=padding, bias=True)
      conv.bias.data.zero_()
      nn.init.xavier_uniform(conv.weight.data, nn.init.calculate_gain('relu'))
      layers.append(conv)
      layers.append(nn.ReLU(inplace=True))

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
               batchnorm=True):
    super(SkipAutoencoder, self).__init__()
    ds_layers = []
    widths = []

    self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')

    # ds_layers.append(SkipAutoencoderDownsample(ninputs, ksize, w, batchnorm=False))
    prev_w = ninputs
    for d in range(depth):
      ds_layers.append(ConvBNRelu(prev_w, ksize, width, batchnorm=batchnorm, stride=2))
      prev_w = width
      # TODO: no bn for layer 1?

    us_layers = []
    for d in range(depth-1, -1, -1):
      if d == 0:
        w_input = width+ninputs
      else:
        w_input = width*2
      us_layers.append(ConvBNRelu(w_input, ksize, width, batchnorm=batchnorm))

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
      x = self.upsampler(x)
      prev = data.pop()
      x = th.cat((x, prev), 1)
      x = l(x)

    x = self.prediction(x)
    return x
