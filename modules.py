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
        nn.init.xavier_uniform(fc.weight.data, nn.init.calculate_gain('relu'))
        raise ValueError("check batchnorm correctness in FC torchlib")
        bn = nn.BatchNorm1d(width)
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

    if depth > 1:
      _in = width
    else:
      _in = ninputs
    fc = nn.Linear(_in, noutputs)
    fc.bias.data.zero_()
    nn.init.xavier_uniform(fc.weight.data)
    layers.append(fc)

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    x = self.net(x)
    return x


class ConvChain(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, depth=3, stride=1,
               pad=True, batchnorm=False, output_type="linear"):
    super(ConvChain, self).__init__()

    assert depth > 0

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
      layers.append(
          ConvBNRelu(
            _in, ksize, width, batchnorm=batchnorm, padding=padding, 
            stride=stride))

    # Last layer
    if depth > 1:
      _in = width
    else:
      _in = ninputs

    conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
    conv.bias.data.zero_()
    nn.init.xavier_uniform(
        conv.weight.data, nn.init.calculate_gain(output_type))
    layers.append(conv)

    # Rename layers
    for im, m in enumerate(layers):
      if im == len(layers)-1:
        name = "prediction"
      else:
        name = "layer_{}".format(im)
      self.add_module(name, m)

    if output_type == "linear":
      pass
    elif output_type == "relu":
      self.add_module("output_activation", nn.ReLU(inplace=True))
    elif output_type == "sigmoid":
      self.add_module("output_activation", nn.Sigmoid())
    elif output_type == "tanh":
      self.add_module("output_activation", nn.Tanh())
    else:
      raise ValueError("Unknon output type '{}'".format(output_type))

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x


class ConvBNRelu(nn.Module):
  def __init__(self, ninputs, ksize, noutputs, batchnorm=False, stride=1, padding=0):
    super(ConvBNRelu, self).__init__()
    if batchnorm:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding, bias=False)
      bn = nn.BatchNorm2d(noutputs)
      bn.bias.data.zero_()
      bn.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, bn, nn.ReLU(inplace=True))
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding)
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

    widths = []
    prev_w = ninputs
    for d in range(depth):
      if grow_width:
        _out = min(width*(2**d), max_width)
      else:
        _out = width
      ds_layers.append(ConvBNRelu(prev_w, ksize, _out, batchnorm=batchnorm,
                                  stride=2, padding=ksize//2))
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
      us_layers.append(ConvBNRelu(_in, ksize, _out, batchnorm=batchnorm, 
                                  padding=ksize//2))

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


class Autoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3, num_convs=2, 
               max_width=512, increase_factor=1.0, batchnorm=False, output_type="linear",
               pooling="conv"):
    super(Autoencoder, self).__init__()
    next_level = None
    for lvl in range(num_levels-1, -1, -1):
      w = min(int(width*(increase_factor)**(lvl)), max_width)
      w2 = min(int(width*(increase_factor)**(lvl+1)), max_width)
      n_in = w
      n_out = w
      n_ds = w2
      n_us = w2
      o_type = "relu"
      if lvl == 0:
        n_in = ninputs
        n_out = noutputs
        o_type = output_type
      elif lvl == num_levels-1:
        n_ds = None
        n_us = None

      # print lvl, n_in, n_ds, n_us, n_out
      next_level = AutoencoderLevel(n_in, n_out, next_level=next_level, num_ds=n_ds, num_us=n_us,
                      ksize=ksize, width=w, num_convs=num_convs, 
                      output_type=o_type, batchnorm=batchnorm, pooling=pooling)
    self.add_module("net", next_level)

  def forward(self, x):
    return self.net(x)


class AutoencoderLevel(nn.Module):
  def __init__(self, num_inputs, num_outputs, next_level=None,
               num_ds=None, num_us=None,
               ksize=3, width=64, num_convs=2, output_type="linear",
               batchnorm=True, pooling="conv"):
    super(AutoencoderLevel, self).__init__()

    self.is_last = (next_level is None)

    if self.is_last:
      self.left = ConvChain(num_inputs, num_outputs, ksize=ksize, width=width,
                            depth=num_convs, stride=1, pad=True, 
                            batchnorm=batchnorm, output_type=output_type)
    else:
      assert num_ds is not None
      assert num_us is not None

      self.left = ConvChain(
          num_inputs, width, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, batchnorm=batchnorm,
          output_type="relu")
      if pooling == "conv":
        self.downsample = nn.Sequential(
            nn.Conv2d(width, num_ds, stride=2, kernel_size=4, padding=1),
            nn.ReLU(inplace=True)
            )
      elif pooling == "max":
        self.downsample = nn.MaxPool2d(2, 2)
      else:
        raise ValueError("unknown pooling'{}'".format(pooling))

      self.next_level = next_level
      self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
      self.right = ConvChain(
          num_us + width, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, batchnorm=batchnorm,
          output_type=output_type)

  def forward(self, x):
    if self.is_last:
      return self.left(x)
    else:
      left = self.left(x)
      ds = self.downsample(left)
      next_level = self.next_level(ds)
      us = self.upsample(next_level)
      concat = th.cat([us, left], 1)
      output = self.right(concat)
      return output
