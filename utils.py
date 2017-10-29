import logging
import torch as th
from torch.autograd import Variable

log = logging.getLogger(__name__)


def make_variable(d, cuda=True):
  ret = {}
  for k in d.keys():
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

