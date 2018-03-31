import time
import logging

from tqdm import tqdm

import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

import torchlib.utils as utils
import torchlib.callbacks as callbacks

class Trainer(object):
  """docstring for Trainer"""

  class Parameters(object):
    def __init__(self, optimizer=optim.Adam, 
                 batch_size=1, lr=1e-4, wd=0, viz_smoothing=0.999,
                 viz_step=100,
                 checkpoint_interval=60):
      self.batch_size = batch_size
      self.lr = lr
      self.wd = wd
      self.optimizer = optimizer
      self.viz_smoothing = viz_smoothing
      self.viz_step = viz_step
      self.checkpoint_interval = checkpoint_interval

  def cuda(self, b):
    self._cuda = b
    if b:
      self.model.cuda()
      for l in self.criteria:
        self.criteria[l].cuda()
      self.log.debug("swich to cuda")

  def __init__(self, trainset, model, criteria, output=None,
               model_params=None,
               params=None, metrics={}, cuda=False,
               callbacks=[callbacks.LossCallback()], valset=None, verbose=False):

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
    self.log_keys = list(criteria.keys())
    self.log_keys.append("loss")

    if metrics is not None:
      self.log_keys += list(self.metrics.keys())

    self.cuda(cuda)

    self.ema = utils.ExponentialMovingAverage(
        self.log_keys, alpha=self.params.viz_smoothing)
    self.averager = utils.Averager(self.log_keys)

    self.callbacks = callbacks

    self.optimizer = self.params.optimizer(
        self.model.parameters(),
        lr=self.params.lr, 
        weight_decay=self.params.wd)

    if output is not None:
      self.checkpointer = utils.Checkpointer(
          output, self.model, self.optimizer,
          meta_params={"model": model_params},
          interval=self.params.checkpoint_interval)
    else:
      self.checkpointer = None

    self.train_loader = DataLoader(
      self.trainset, batch_size=self.params.batch_size, 
      shuffle=True, num_workers=4)

    if self.valset is not None:
      self.val_loader = DataLoader(
          self.valset, batch_size=self.params.batch_size,
          shuffle=True, num_workers=0, 
          drop_last=True)  # so we have a fixed batch size for averaging in val
    else:
      self.val_loader = None
  
    self.log.debug("Model: {}\n".format(model))
    self.log.debug("Parameters to train:")
    for n, p in model.named_parameters():
      self.log.debug('  - {}'.format(n))

    if self.checkpointer is None:
      self.log.warn("No checkpointer provided, progress will not be saved.")

    self._set_model()
  
  def _on_epoch_begin(self):
    self.log.debug("Epoch begins")
    for c in self.callbacks:
      c.on_epoch_begin(self.epoch)

  def _on_epoch_end(self, logs):
    if logs is None:
      return

    self.log.debug("Epoch ends")
    for c in self.callbacks:
      c.on_epoch_end(self.epoch, logs)

  def _on_batch_end(self, batch_id, num_batches, logs):
    self.log.debug("Batch ends")
    for c in self.callbacks:
      c.on_batch_end(batch_id, num_batches, logs)

  def _train_one_epoch(self, num_epochs):
    self.model.train(True)
    with tqdm(total=len(self.train_loader), unit=' batches') as pbar:
      pbar.set_description("Epoch {}/{}".format(
        self.epoch+1, num_epochs if num_epochs > 0 else "--"))

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
        for k in self.metrics.keys():
          m = self.metrics[k](batch_v, output)
          self.ema.update(k, m.cpu().data.item())

        loss.backward()

        self.optimizer.step()

        logs = {k: self.ema[k] for k in self.log_keys}
        pbar.set_postfix(logs)

        if pbar.n % self.params.viz_step == 0:
          self._on_batch_end(batch_id, len(self.train_loader), logs)

        pbar.update(1)

        if self.checkpointer is not None:
          self.checkpointer.periodic_checkpoint(self.epoch)

  def _set_model(self):
    if self.checkpointer:
      chkpt_name, epoch = self.checkpointer.load_latest()
      if chkpt_name is None:
        self.log.info("Starting training from scratch")
      else:
        self.log.info("Resuming from latest checkpoint {}.".format(chkpt_name))
    else:
      epoch = 0
    self.epoch = epoch

  def override_parameters(self, checkpoint):
    if checkpoint and self.checkpointer:
      self.log.info("Overriding parameters:")
      names = self.checkpointer.override_params(args.checkpoint)
      for n in names:
        self.log.info("  - {}".format(n))

  def train(self, num_epochs=-1):
    best_val_loss = None
    try:
      while True:
        # Training
        self._on_epoch_begin() 
        self._train_one_epoch(num_epochs)

        # Validation
        val_loss, val_logs = self._run_validation(num_epochs)
        if best_val_loss is None:
          best_val_loss = val_loss
        if self.checkpointer and val_loss and val_loss <= best_val_loss:
          self.checkpointer.save_best(self.epoch)

        self.epoch += 1

        self._on_epoch_end(val_logs) 

        if num_epochs > 0 and self.epoch >= num_epochs:
          self.log.info("Ending training at epoch {} of {}".format(self.epoch, num_epochs))

    except KeyboardInterrupt:
      self.log.info("training interrupted")

  def _run_validation(self, num_epochs):
    count = self.params.batch_size
    logs = None

    if self.val_loader is None:
      return None, logs

    with th.no_grad():
      self.model.train(False)
      self.averager.reset()
      with tqdm(total=len(self.val_loader), unit=' batches') as pbar:
        pbar.set_description("Epoch {}/{} (val)".format(
          self.epoch+1, num_epochs if num_epochs > 0 else "--"))
        for batch_id, batch in enumerate(self.val_loader):
          batch_v = utils.make_variable(batch, cuda=self._cuda)
          output = self.model(batch_v)

          # Compute all losses
          c_out = []
          for k in self.criteria.keys():
            c_out.append(self.criteria[k](batch_v, output))
          loss = sum(c_out)
          self.averager.update("loss", loss.cpu().data.item(), count)

          # Compute all metrics
          for k in self.metrics.keys():
            m = self.metrics[k](batch_v, output)
            self.averager.update(k, m.cpu().data.item())

          pbar.update(1)

          logs = {k: self.averager[k] for k in self.log_keys}
          pbar.set_postfix(logs)

        return self.averager["loss"], logs
