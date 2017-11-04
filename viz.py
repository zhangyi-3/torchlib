import visdom
import numpy as np
from skimage.color import hsv2rgb

class Visualizer(object):
  def __init__(self, port=8097, env="main"):
    self.vis = visdom.Visdom(port=port, env=env)


class ScalarVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(ScalarVisualizer, self).__init__(port=port, env=env)
    self.name = name
    self.time = []
    self.value = []
    self.plot_update = None

  def update(self, t, v, legend=None):
    self.time.append(t)
    self.value.append(v)

    plot_update = None
    if self.vis.win_exists(self.name, env=self.vis.env):
      plot_update = True

    self.vis.line(
        X=np.array(self.time),
        Y=np.array(self.value),
        update=plot_update,
        opts={
          'title': "{} over time".format(self.name),
          'xlabel': 'epoch',
          'ylabel': self.name,
          'legend': legend
          },
        win=self.name)


class ImageVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(ImageVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, image, caption=None):
    self.vis.images(
        image,
        opts={
          'title': "{}".format(self.name),
          'jpgquality': 100,
          'caption': caption,
          },
        win=self.name)


class BatchVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(BatchVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, images, per_row=8, caption=None):
    self.vis.images(
        images,
        nrow=per_row,
        opts={
          'title': "{}".format(self.name),
          'jpgquality': 100,
          'caption': caption,
          },
        win=self.name)


class ScatterVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(ScatterVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, x, title="", xlabel="x", ylabel="y",
             color=None):
    plot_update = None
    if self.vis.win_exists(self.name, env=self.vis.env):
      plot_update = False

    plot_update = None
    self.vis.scatter(
        X=x,
        update=plot_update,
        opts={
          'title': title,
          'xlabel': xlabel,
          'ylabel': ylabel,
          'markercolor': color,
          },
        win=self.name)
