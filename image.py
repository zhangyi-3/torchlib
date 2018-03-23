import numpy as np

def crop_like(src, tgt):
  src_sz = np.array(src.shape)
  tgt_sz = np.array(tgt.shape)
  crop = (src_sz[2:4]-tgt_sz[2:4]) // 2
  if (crop > 0).any():
    return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1], ...]
  else:
    return src

def read_pfm(path):
  with open(path, 'rb') as fid:
    identifier = fid.readline().strip()
    if identifier == b'PF':  # color
      nchans = 3
    elif identifier == b'Pf':  # gray
      nchans = 1
    else:
      raise ValueError("Unknown PFM identifier {}".format(identifier))

    dimensions = fid.readline().strip()
    width, height = [int(x) for x in dimensions.split()]
    endianness = fid.readline().strip()

    data = np.fromfile(fid, dtype=np.float32, count=width*height*nchans)
    data = np.reshape(data, (height, width, nchans))

    return data
  
