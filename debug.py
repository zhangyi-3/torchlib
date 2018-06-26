import torch as th
import torchlib.viz as viz
import numpy as np

def tensor(t, key="none", env="debug"):
  t = t.detach()
  v = viz.BatchVisualizer(key, env=env)
  bs, c, h, w = t.shape
  if c != 1 and c != 3:
    t = t.view(bs*c, 1, h, w)
  mu = t.mean()
  std = t.std()
  t = (t-mu) / (2*std + 1e-8)
  t = th.clamp(0.5*(t+1), 0, 1)
  v.update(t.cpu().numpy(), caption="{} {:.2f} ({:.2f})".format(key, mu, std))

def histogram(t, bins=10, key="none", env="debug"):
  v = viz.HistogramVisualizer(key, env=env)
  v.update(t.detach().cpu().numpy(), numbins=bins)

def scatter(x, y, key="none", env="debug"):
  v = viz.ScatterVisualizer(key, env=env)
  xx = th.cat([x.view(-1, 1), y.view(-1, 1)], 1).detach().cpu().numpy()
  v.update(xx)

def line(x, y, key="none", ylog=False, env="debug"):
  opts = {}
  if ylog:
    opts["ytype"] = "log"
  v = viz.ScalarVisualizer(key, env=env, opts=opts)
  xx = np.ravel(x)
  yy = np.ravel(y)
  for i in range(xx.size):
    v.update(xx[i], yy[i])
