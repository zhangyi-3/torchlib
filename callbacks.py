
class Callback(object):
  def __init__(self):
    pass

  def on_epoch_begin(self, epoch):
    pass

  def on_epoch_end(self, epoch, logs, batch_data=None):
    pass

  def on_batch_end(self, batch, logs):
    pass

class LossCallback(Callback):
  pass
