import logging
import torch as th
from torch.autograd import Variable
import time

log = logging.getLogger(__name__)

import os
import re

def make_variable(d, cuda=True):
  ret = {}
  for k in d.keys():
    if "Tensor" not in type(d[k]).__name__:
      ret[k] = d[k]
      continue
    if cuda:
      ret[k] = Variable(d[k].cuda())
    else:
      ret[k] = Variable(d[k])
  return ret


def save(checkpoint, model, params, optimizer, step):
  log.info("saving checkpoint {} at step {}".format(checkpoint, step))
  th.save({
    'model_state': model.state_dict(),
    'params': params,
    'optimizer': optimizer.state_dict(),
    'step': step,
    } , checkpoint)


class Checkpointer(object):
  def __init__(self, directory, model, optimizer, 
               max_save=5,
               interval=-1,
               extra_params=None,
               filename='ckpt.pth.tar', verbose=False):
    """
    If interval > 0, checkpoints every "interval seconds".
    """

    if directory.startswith('~'):
        directory = os.path.expanduser(directory)
    self.model = model
    self.optimizer = optimizer
    self.max_save = max_save
    self.directory = directory
    self.filename = filename
    self.verbose = verbose
    self.extra_params = extra_params
    self.interval = interval

    if self.interval > 0:
      self.last_checkpoint_time = time.time()


    self.reg = re.compile(r".*\.pth\.tar")
    all_checkpoints = [f for f in os.listdir(self.directory) if self.reg.match(f)]

    reg_epoch = re.compile(r"epoch.*\.pth\.tar")
    reg_periodic = re.compile(r"periodic.*\.pth\.tar")
    self.old_epoch_files = sorted([c for c in all_checkpoints if reg_epoch.match(c)])
    self.old_timed_files = sorted([c for c in all_checkpoints if reg_periodic.match(c)])

  def load_latest(self):
    all_checkpoints = [f for f in os.listdir(self.directory) if self.reg.match(f)]
    if len(all_checkpoints) == 0:
      return None, 0
    mtimes = []
    for f in all_checkpoints:
      mtimes.append(os.path.getmtime(os.path.join(self.directory, f)))

    mf = sorted(zip(mtimes, all_checkpoints))
    for m, f in reversed(mf):
      try:
        e = self.load_checkpoint(os.path.join(self.directory, f))
        return f, e
      except:
        print "could not load latest checkpoint {}, moving on.".format(f)
    return None, -1

  def save_checkpoint(self, epoch, filename, is_best=False):
    th.save({ 
        'epoch': epoch + 1,
        'state_dict': self.model.state_dict(),
        'optimizer' : self.optimizer.state_dict(),
        'extra_params': self.extra_params,
        }, os.path.join(self.directory, filename))
    if is_best:
      shutil.copyfile(filename, os.path.join(self.directory, 'best.pth.tar'))

  def periodic_checkpoint(self, epoch):
    now = time.time()
    if self.interval <= 0 or (
      now - self.last_checkpoint_time < self.interval):
      return
    self.last_checkpoint_time = now

    filename = 'periodic_{}.pth.tar'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    self.save_checkpoint(epoch, filename)
    if self.max_save > 0:
      if len(self.old_timed_files) >= self.max_save:
        self.delete_checkpoint(self.old_timed_files[0])
        self.old_timed_files = self.old_timed_files[1:]
      self.old_timed_files.append(filename)

  def delete_checkpoint(self, filename):
    try:
      os.remove(os.path.join(self.directory, filename))
    except:
      print "exception in chekpoint deletion."
      pass

  def load_checkpoint(self, filename):
    chkpt = th.load(filename)
    self.model.load_state_dict(chkpt["state_dict"])
    self.optimizer.load_state_dict(chkpt["optimizer"])
    return chkpt["epoch"]

  def on_epoch_end(self, epoch):
    filename = 'epoch_{:03d}.pth.tar'.format(epoch+1)
    if self.verbose > 0:
      print('\nEpoch %i: saving model to %s' % (epoch+1, file))
    self.save_checkpoint(epoch, filename)
    if self.max_save > 0:
      if len(self.old_epoch_files) >= self.max_save:
        self.delete_checkpoint(self.old_epoch_files[0])
        self.old_epoch_files = self.old_epoch_files[1:]
      self.old_epoch_files.append(filename)


class ExponentialMovingAverage(object):
  def __init__(self, keys, alpha=0.99):
    self.first_update = {k: True for k in keys}
    self.alpha = alpha
    self.values = {k: 0 for k in keys}

  def __getitem__(self, key):
    return self.values[key]

  def update(self, key, value):
    if self.first_update[key]:
      self.values[key] = value
      self.first_update[key] = False
    else:
      self.values[key] = self.values[key]*self.alpha + value*(1.0-self.alpha)


class Averager(object):
  def __init__(self, keys):
    self.values = {k: 0.0 for k in keys}
    self.counts = {k: 0 for k in keys}

  def __getitem__(self, key):
    return self.values[key] * 1.0/self.counts[key]

  def reset(self):
    for k in self.values.keys():
      self.values[k] = 0.0
      self.counts[k] = 0

  def update(self, key, value, count=1):
    self.values[key] += value
    self.counts[key] = count
