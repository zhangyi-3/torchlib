from collections import OrderedDict

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
               pad=True, batchnorm=False, output_type="linear", activation="relu"):
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
            stride=stride, activation=activation))

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
    elif output_type == "leaky_relu":
      self.add_module("output_activation", nn.LeakyReLU(inplace=True))
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
  def __init__(self, ninputs, ksize, noutputs, batchnorm=False, stride=1, padding=0,
               activation="relu"):
    super(ConvBNRelu, self).__init__()
    if activation == "relu":
      act_fn = nn.ReLU
    elif activation == "leaky_relu":
      act_fn = nn.LeakyReLU
    else:
      raise NotImplemented

    if batchnorm:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding, bias=False)
      bn = nn.BatchNorm2d(noutputs)
      bn.bias.data.zero_()
      bn.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, bn, act_fn(inplace=True))
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding)
      conv.bias.data.zero_()
      self.layer = nn.Sequential(conv, act_fn(inplace=True))

    nn.init.xavier_uniform(conv.weight.data, nn.init.calculate_gain(activation))

  def forward(self, x):
    out = self.layer(x)
    return out


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


class RecurrentAutoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3, num_convs_pre_hidden=1, num_convs=2, 
               max_width=512, increase_factor=1.0, batchnorm=False, output_type="linear",
               pooling="max"):
    super(RecurrentAutoencoder, self).__init__()

    self.num_levels = num_levels
    self.width = width
    self.max_width = max_width
    self.increase_factor = increase_factor

    next_level = None
    for lvl in range(num_levels-1, -1, -1):
      w = min(int(width*(increase_factor)**(lvl)), max_width)
      w2 = min(int(width*(increase_factor)**(lvl+1)), max_width)
      n_out = w
      n_us = w2
      o_type = "relu"
      if lvl == 0:
        n_in = ninputs
        n_out = noutputs
        o_type = output_type
      else:
        w0 = min(int(width*(increase_factor)**(lvl-1)), max_width)
        n_in = w0

      if lvl == num_levels-1:
        n_us = None

      next_level = RecurrentAutoencoderLevel(n_in, n_out, next_level=next_level, num_us=n_us,
                      ksize=ksize, width=w, num_convs=num_convs, num_convs_pre_hidden=num_convs_pre_hidden,
                      output_type=o_type, batchnorm=batchnorm, pooling=pooling)

    self.add_module("net", next_level)

  def forward(self, x, state):
    output, new_state = self.net(x, state)
    return output, new_state

  def get_init_state(self, ref_input):
    state = []
    bs, ci, h, w = ref_input.shape[:4]
    for lvl in range(self.num_levels):
      chans = min(int(self.width*(self.increase_factor)**(lvl)), self.max_width)
      state_lvl = ref_input.new()
      state_lvl.resize_(bs, chans, h, w)
      state_lvl.zero_()
      state.append(state_lvl)
      h /= 2
      w /= 2
    state.reverse()
    return state


class RecurrentAutoencoderLevel(nn.Module):
  def __init__(self, num_inputs, num_outputs, next_level=None,
               num_us=None,
               ksize=3, width=64, num_convs=2, num_convs_pre_hidden=1,
               output_type="linear",
               batchnorm=True, pooling="max"):
    super(RecurrentAutoencoderLevel, self).__init__()

    self.is_last = (next_level is None)

    if self.is_last:
      self.pre_hidden = ConvChain(
          num_inputs, width, ksize=ksize, width=width,
          depth=num_convs_pre_hidden, stride=1, pad=True, batchnorm=batchnorm,
          output_type="leaky_relu", activation="leaky_relu")
      self.left = ConvChain(width+width, num_outputs, ksize=ksize, width=width,
                            depth=num_convs, stride=1, pad=True, 
                            batchnorm=batchnorm, output_type=output_type)
    else:
      assert num_us is not None

      self.pre_hidden = ConvChain(
          num_inputs, width, ksize=ksize, width=width,
          depth=num_convs_pre_hidden, stride=1, pad=True, batchnorm=batchnorm,
          output_type="leaky_relu", activation="leaky_relu")
      self.left = ConvChain(
          width+width, width, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, batchnorm=batchnorm,
          output_type="leaky_relu", activation="leaky_relu")

      if pooling == "conv":
        self.downsample = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(width, width, stride=2, kernel_size=4, padding=1)),
            ("relu", nn.ReLU(inplace=True))
          ])
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

  def forward(self, x, state):
    this_state = state.pop()
    if self.is_last:
      pre_hidden = self.pre_hidden(x)
      new_state = self.left(th.cat([pre_hidden, this_state], 1))  # this is also the new hidden state
      return new_state, [new_state]
    else:
      pre_hidden = self.pre_hidden(x)
      new_state = self.left(th.cat([pre_hidden, this_state], 1))  # this is also the new hidden state
      ds = self.downsample(new_state)
      next_level, next_state = self.next_level(ds, state)
      next_state.append(new_state)
      us = self.upsample(next_level)
      concat = th.cat([us, new_state], 1)
      output = self.right(concat)
      return output, next_state
