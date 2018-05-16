import torch as th
import torchlib.viz as viz
import numpy as np

def tensor(t, key="none"):
  t = t.detach()
  v = viz.BatchVisualizer(key, env="debug")
  bs, c, h, w = t.shape
  mu = t.mean()
  std = t.std()
  t = (t-mu) / (2*std + 1e-8)
  t = th.clamp(0.5*(t+1), 0, 1)
  v.update(t.cpu().numpy(), caption="{} {:.2f} ({:.2f})".format(key, mu, std))

def histogram(t, bins=10, key="none"):
  v = viz.HistogramVisualizer(key, env="debug")
  v.update(t.detach().cpu().numpy(), numbins=bins)

def scatter(x, y, key="none"):
  v = viz.ScatterVisualizer(key, env="debug")
  xx = th.cat([x.view(-1, 1), y.view(-1, 1)], 1).detach().cpu().numpy()
  v.update(xx)

def line(x, y, key="none", ylog=False):
  opts = {}
  if ylog:
    opts["ytype"] = "log"
  v = viz.ScalarVisualizer(key, env="debug", opts=opts)
  xx = np.ravel(x)
  yy = np.ravel(y)
  for i in range(xx.size):
    v.update(xx[i], yy[i])
