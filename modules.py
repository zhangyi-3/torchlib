from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torchlib.image import crop_like

import rendernet.utils as rutils

class FullyConnected(nn.Module):
  def __init__(self, ninputs, noutputs, width=32, depth=3, 
               normalize=False, dropout=False):
    super(FullyConnected, self).__init__()

    if dropout:
      raise NotImplemented

    layers = []
    for d in range(depth-1):
      if d == 0:
        _in = ninputs
      else:
        _in = width
      if normalize:
        fc = nn.Linear(_in, width, bias=False)
        nn.init.xavier_uniform_(fc.weight.data, nn.init.calculate_gain('relu'))
        raise ValueError("check batchnorm correctness in FC torchlib")
        bn = nn.BatchNorm1d(width)
        bn.bias.data.zero_()
        bn.weight.data.fill_(1.0)
        layers.append(fc)
        layers.append(bn)
      else:
        fc = nn.Linear(_in, width, bias=True)
        fc.bias.data.zero_()
        nn.init.xavier_uniform_(fc.weight.data, nn.init.calculate_gain('relu'))
        layers.append(fc)
      layers.append(nn.ReLU(inplace=True))

    if depth > 1:
      _in = width
    else:
      _in = ninputs
    fc = nn.Linear(_in, noutputs)
    fc.bias.data.zero_()
    nn.init.xavier_uniform_(fc.weight.data)
    layers.append(fc)

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    x = self.net(x)
    return x


class ConvChain(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, depth=3, stride=1,
               pad=True, normalize=False, normalization_type="batch", 
               output_type="linear", 
               activation="relu"):
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
            _in, ksize, width, normalize=normalize, normalization_type="batch", padding=padding, 
            stride=stride, activation=activation))

    # Last layer
    if depth > 1:
      _in = width
    else:
      _in = ninputs

    conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
    conv = nn.utils.weight_norm(conv)  # TODO
    conv.bias.data.zero_()
    if output_type == "elu" or output_type == "softplus":
      nn.init.xavier_uniform_(
          conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(
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
    elif output_type == "elu":
      self.add_module("output_activation", nn.ELU())
    elif output_type == "softplus":
      self.add_module("output_activation", nn.Softplus())
    else:
      raise ValueError("Unknon output type '{}'".format(output_type))

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x


class ConvBNRelu(nn.Module):
  def __init__(self, ninputs, ksize, noutputs, normalize=False, 
               normalization_type="batch", stride=1, padding=0,
               activation="relu"):
    super(ConvBNRelu, self).__init__()
    if activation == "relu":
      act_fn = nn.ReLU
    elif activation == "leaky_relu":
      act_fn = nn.LeakyReLU
    elif activation == "tanh":
      act_fn = nn.Tanh
    elif activation == "elu":
      act_fn = nn.ELU
    else:
      raise NotImplemented

    if normalize:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding, bias=False)
      if normalization_type == "batch":
        nrm = nn.BatchNorm2d(noutputs)
      elif normalization_type == "instance":
        nrm = nn.InstanceNorm2D(noutputs)
      else:
        raise ValueError("Unkown normalization type {}".format(normalization_type))
      nrm.bias.data.zero_()
      nrm.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, nrm, act_fn())
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding)
      conv = nn.utils.weight_norm(conv)  # TODO
      conv.bias.data.zero_()
      self.layer = nn.Sequential(conv, act_fn())

    if activation == "elu":
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain(activation))

  def forward(self, x):
    out = self.layer(x)
    return out


class Autoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3, 
               num_convs=2, max_width=512, increase_factor=1.0, 
               normalize=False, normalization_type="batch", 
               output_type="linear",
               activation="relu", pooling="max"):
    super(Autoencoder, self).__init__()
    

    next_level = None
    for lvl in range(num_levels-1, -1, -1):
      n_in = min(int(width*(increase_factor)**(lvl-1)), max_width)
      w = min(int(width*(increase_factor)**(lvl)), max_width)
      n_us = min(int(width*(increase_factor)**(lvl+1)), max_width)
      n_out = w
      o_type = activation

      if lvl == 0:
        n_in = ninputs
        o_type = output_type
        n_out = noutputs
      elif lvl == num_levels-1:
        n_us = None

      next_level = AutoencoderLevel(
          n_in, n_out, next_level=next_level, num_us=n_us,
          ksize=ksize, width=w, num_convs=num_convs, 
          output_type=o_type, normalize=normalize, 
          normalization_type=normalization_type,
          activation=activation, pooling=pooling)

    self.add_module("net", next_level)

  def forward(self, x):
    return self.net(x)


class AutoencoderLevel(nn.Module):
  def __init__(self, num_inputs, num_outputs, next_level=None,
               num_us=None,
               ksize=3, width=64, num_convs=2, output_type="linear",
               normalize=True, normalization_type="batch", pooling="max",
               activation="relu"):
    super(AutoencoderLevel, self).__init__()

    self.is_last = (next_level is None)

    if self.is_last:
      self.left = ConvChain(
          num_inputs, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, 
          normalize=normalize, normalization_type=normalization_type,
          output_type=output_type)
    else:
      assert num_us is not None

      self.left = ConvChain(
          num_inputs, width, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, normalize=normalize,
          normalization_type=normalization_type,
          output_type=activation, activation=activation)
      if pooling == "max":
        self.downsample = nn.MaxPool2d(2, 2)
      elif pooling == "average":
        self.downsample = nn.AvgPool2d(2, 2)
      elif pooling == "conv":
        self.downsample = nn.Conv2d(width, width, 2, stride=2)
      else:
        raise ValueError("unknown pooling'{}'".format(pooling))

      self.next_level = next_level
      self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
      self.right = ConvChain(
          num_us + width, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, normalize=normalize,
          normalization_type=normalization_type,
          output_type=output_type)

  def forward(self, x):
    left = self.left(x)
    if self.is_last:
      return left

    ds = self.downsample(left)
    next_level = self.next_level(ds)
    us = self.upsample(next_level)
    concat = th.cat([us, left], 1)
    output = self.right(concat)
    return output


class RecurrentAutoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3, 
               num_convs_state=1, num_convs=2, 
               max_width=512, increase_factor=1.0, 
               normalize=False, normalization_type="batch",
               activation="leaky_relu", recurrent_activation="leaky_relu",
               output_type="linear", pooling="max", pad=True,
               temporal_ksize=None):
    super(RecurrentAutoencoder, self).__init__()

    if not pad:
      raise ValueError("rnn with no padding is not tested!")

    self.num_levels = num_levels
    self.width = width
    self.max_width = max_width
    self.increase_factor = increase_factor
    self.ksize = ksize
    self.num_convs = num_convs
    self.num_convs_state = num_convs_state
    self.pad = pad

    next_level = None
    for lvl in range(num_levels-1, -1, -1):
      n_in = min(int(width*(increase_factor)**(lvl-1)), max_width)
      w = min(int(width*(increase_factor)**(lvl)), max_width)
      n_us = min(int(width*(increase_factor)**(lvl+1)), max_width)
      n_out = w
      o_type = activation

      if lvl == 0:
        n_in = ninputs
        n_out = noutputs
        o_type = output_type
      elif lvl == num_levels-1:
        n_us = None

      next_level = RecurrentAutoencoderLevel(
          n_in, n_out, next_level=next_level, num_us=n_us,
          ksize=ksize, width=w, num_convs=num_convs, num_convs_state=num_convs_state,
          recurrent_activation=recurrent_activation,
          output_type=o_type, normalize=normalize, activation=activation,
          normalization_type=normalization_type, pooling=pooling, pad=pad, lvl=lvl,
          temporal_ksize=temporal_ksize)

    self.add_module("net", next_level)

  def forward(self, x, state, encoder_only=False):
    output, new_state = self.net(x, state, encoder_only)
    return output, new_state

  def get_init_state(self, ref_input):
    state = []
    bs, ci, h, w = ref_input.shape[:4]
    for lvl in range(self.num_levels):
      chans = min(int(self.width*(self.increase_factor)**(lvl)), self.max_width)
      state_lvl = ref_input.data.new()
      state_lvl.resize_(bs, chans, int(h), int(w))
      state_lvl.zero_()
      state_lvl = Variable(state_lvl)
      state.append(state_lvl)
      h /= 2
      w /= 2
    state.reverse()
    return state


class RecurrentAutoencoderLevel(nn.Module):
  def __init__(self, num_inputs, num_outputs, next_level=None,
               num_us=None,
               ksize=3, width=64, num_convs=2, num_convs_state=1,
               activation="leaky_relu", output_type="linear", recurrent_activation="leaky_relu",
               normalize=True, normalization_type="batch", pooling="max",
               pad=True, lvl=-1, temporal_ksize=None):
    super(RecurrentAutoencoderLevel, self).__init__()

    self.lvl = lvl
    self.debug = rutils.DebugFeatureVisualizer(
        ["output_{}".format(lvl)], period=10)

    if temporal_ksize is None:
      temporal_ksize=ksize

    if not pad:
      raise ValueError("rnn with no padding is not tested!")

    self.is_last = (next_level is None)
    self.pad = pad
    self.num_convs = num_convs
    self.ksize = ksize

    n_left_outputs = width
    if self.is_last:
      n_left_outputs = num_outputs

    self.pre_hidden = ConvChain(
        num_inputs, width, ksize=ksize, width=width,
        depth=num_convs, stride=1, pad=pad, normalize=normalize,
        normalization_type=normalization_type,
        output_type=activation, activation=activation)

    self.left = ConvChain(
        width + width, n_left_outputs, ksize=temporal_ksize, width=width,
        depth=num_convs_state, stride=1, pad=pad, normalize=normalize,
        normalization_type=normalization_type,
        output_type=recurrent_activation, activation=activation)

    if self.is_last:
      self.right = ConvChain(
          width, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=pad, normalize=normalize,
          normalization_type=normalization_type,
          activation=activation,
          output_type=output_type)
    else:
      assert num_us is not None

      if pooling == "max":
        self.downsample = nn.MaxPool2d(2, 2)
      elif pooling == "average":
        self.downsample = nn.AvgPool2d(2, 2)
      elif pooling == "conv":
        self.downsample = nn.Conv2d(n_left_outputs, n_left_outputs, 2, stride=2)
      else:
        raise ValueError("unknown pooling'{}'".format(pooling))

      self.next_level = next_level
      self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
      self.pre_right = ConvChain(
          num_us, width, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=pad, normalize=normalize,
          normalization_type=normalization_type,
          activation=activation,
          output_type=output_type)
      self.right = ConvChain(
          2*width, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=pad, normalize=normalize,
          normalization_type=normalization_type,
          activation=activation,
          output_type=output_type)

  def forward(self, x, state, encoder_only=False):
    this_state = state.pop()
    pre_hidden = self.pre_hidden(x)

    cc = th.cat([pre_hidden, this_state], 1)
    new_state = self.left(cc)  # this is also the new hidden state

    if self.is_last:
      output = self.right(new_state)
      next_state = [new_state]
    else:
      ds = self.downsample(new_state)
      next_level, next_state = self.next_level(ds, state)
      next_state.append(new_state)

      if encoder_only: # only compute the internal recurrent state
        return None, next_state

      us = self.upsample(next_level)
      us_f = self.pre_right(us)
      concat = th.cat([us_f, new_state], 1)
      output = self.right(concat)

    self.debug.update("output_{}".format(self.lvl), output)
    self.debug.step()
    return output, next_state
