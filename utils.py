import logging
import torch as th
from torch.autograd import Variable

log = logging.getLogger(__name__)

import os

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
               max_save=-1,
               filename='ckpt.pth.tar', verbose=False):
    if directory.startswith('~'):
        directory = os.path.expanduser(directory)
    self.model = model
    self.optimizer = optimizer
    self.max_save = max_save
    self.directory = directory
    self.filename = filename
    self.verbose = verbose

    self.old_files = []

  def save_checkpoint(self, epoch, filename, is_best=False):
    th.save({ 
        'epoch': epoch + 1,
        'state_dict': self.model.state_dict(),
        'optimizer' : self.optimizer.state_dict(),
        }, os.path.join(self.directory, filename))
    if is_best:
      shutil.copyfile(filename, os.path.join(self.directory, 'best.pth.tar'))

  def on_epoch_end(self, epoch, logs=None):
    filename = 'epoch_{:03d}.pth.tar'.format(epoch+1)
    if self.verbose > 0:
      print('\nEpoch %i: saving model to %s' % (epoch+1, file))
    self.save_checkpoint(epoch, filename)
    if self.max_save > 0:
      if len(self.old_files) == self.max_save:
        try:
          os.remove(self.old_files[0])
        except:
          pass
        self.old_files = self.old_files[1:]
      self.old_files.append(filename)
