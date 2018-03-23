import time
import logging

from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import DataLoader

import torchlib.utils as utils

class Trainer(object):
  """docstring for Trainer"""

  class Parameters(object):
    def __init__(self, optimizer=optim.Adam, 
                 batch_size=1, lr=1e-4, wd=0, viz_smoothing=0.999):
      self.batch_size = batch_size
      self.lr = lr
      self.wd = wd
      self.optimizer = optimizer
      self.viz_smoothing = viz_smoothing

  def cuda(self, b):
    self._cuda = b
    if b:
      self.model.cuda()
      for l in self.criteria:
        self.criteria[l].cuda()
      self.log.debug("swich to cuda")

  def __init__(self, trainset, model, criteria, checkpointer=None,
               params=None, metrics=None, cuda=False,
               callbacks=None, valset=None, verbose=False):

    self.verbose = verbose
    self.log = logging.getLogger("trainer")
    self.log.setLevel(logging.INFO)
    if self.verbose:
      self.log.setLevel(logging.DEBUG)

    self.model = model
    self.trainset = trainset
    self.valset = valset

    if params is None:
      self.params = Trainer.Parameters()
    else:
      self.params = params

    self.criteria = criteria
    self.metrics = metrics
    self.log_keys = criteria.keys()

    if metrics is not None:
      self.log_keys += metrics.keys()

    self.cuda(cuda)

    self.ema = utils.ExponentialMovingAverage(
        self.log_keys, alpha=self.params.viz_smoothing)
    self.averager = utils.Averager(self.log_keys)

    self.callbacks = callbacks

    self.optimizer = self.params.optimizer(
        self.model.parameters(),
        lr=self.params.lr, 
        weight_decay=self.params.wd)

    self.train_loader = DataLoader(
      self.trainset, batch_size=self.params.batch_size, 
      shuffle=True, num_workers=4)

    if self.valset is not None:
      self.val_loader = DataLoader(
          self.valset, batch_size=self.params.batch_size,
          shuffle=True, num_workers=0)
  
    self.log.debug("Model: {}\n".format(model))
    self.log.debug("Parameters to train:")
    for n, p in model.named_parameters():
      self.log.debug('  - {}'.format(n))

    self.checkpointer = checkpointer
    if self.checkpointer is None:
      self.log.warn("No checkpointer provided, progress will not be saved.")
  
  def on_epoch_begin(self, epoch):
    self.log.debug("Epoch begins")
    # callback.on_epoch_begin(epoch)
    pass

  def on_epoch_end(self, epoch):
    self.log.debug("Epoch ends")

  def _train_one_epoch(self, epoch, num_epochs):
      self.model.train(True)
      with tqdm(total=len(self.train_loader), unit=' batches') as pbar:
        pbar.set_description("Epoch {}/{}".format(
          epoch+1, num_epochs if num_epochs > 0 else "--"))

        for batch_id, batch in enumerate(self.train_loader):
          batch_v = utils.make_variable(batch, cuda=self._cuda)
          self.optimizer.zero_grad()

          start = time.time()
          output = self.model(batch_v)

          elapsed = (time.time() - start)*1000.0
          self.log.debug("Forward {:.1f} ms".format(elapsed))

          # Compute all losses
          c_out = []
          for k in self.criteria.keys():
            c_out.append(self.criteria[k](batch_v, output))
          loss = sum(c_out)
          self.ema.update("loss", loss.cpu().data.item())

          # Compute all metrics
          # rmse = rmse_fn(output, target)
          # self.ema.update("rmse", rmse.cpu().data.item())

          loss.backward()

          self.optimizer.step()

          # if pbar.n % self.viz_step == 0:
          #   self.on_batch_end(batch_id, logs)

          logs = {k: self.ema[k] for k in self.log_keys}
          pbar.set_postfix(logs)
          pbar.update(1)

  def train(self, num_epochs=-1):
    epoch = 0
    try:
      while True:
        # Training
        self.on_epoch_begin(epoch) 
        self._train_one_epoch(epoch, num_epochs)
        self.on_epoch_end(epoch) 

        # Validation
        self._run_validation(epoch, num_epochs)

        # checkpointer.periodic_checkpoint(epoch)
        epoch += 1

        if num_epochs > 0 and epoch >= num_epochs:
          self.log.info("Ending training at epoch {} of {}".format(epoch, num_epochs))
          break

    except KeyboardInterrupt:
      self.log.info("training interrupted")

  def _run_validation(self, epoch, num_epochs):
    if self.val_dataloader is not None:
      with th.no_grad():
        model.train(False)
        self.averager.reset()
        with tqdm(total=len(self.val_loader), unit=' batches') as pbar:
          pbar.set_description("Epoch {}/{} (val)".format(
            epoch+1, num_epochs if num_epochs > 0 else "--"))
          for batch_id, batch in enumerate(self.val_loader):
            batch_v = utils.make_variable(batch, cuda=self._cuda)
            output = self.model(batch_v)

            # Compute all losses
            c_out = []
            for k in self.criteria.keys():
              c_out.append(self.criteria[k](batch_v, output))
            loss = sum(c_out)

            # count = target.shape[0]
    #         val_average.update("loss", loss.data.item(), count)
    #         val_average.update("rmse", rmse.data.item(), count)
    #         pbar.update(1)
    #
    #     logs = {k: val_average[k] for k in log_keys}
    #     pbar.set_postfix(logs)
    #
    #     lowspp = crop_like(batch_v['low_spp'], output)
    #
    #     callback.on_epoch_end(epoch, logs, [lowspp, output, target])
    #
    #     if best_val_loss is None:
    #       best_val_loss = val_average['loss']
    #
    #     if val_average['loss'] <= best_val_loss:
    #       checkpointer.save_best(epoch)

