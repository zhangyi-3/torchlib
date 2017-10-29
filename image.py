import numpy as np

def crop_like(src, tgt):
  src_sz = np.array(src.shape)
  tgt_sz = np.array(tgt.shape)
  crop = (src_sz-tgt_sz)[2:4] // 2
  if (crop > 0).any():
    return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1]]
  else:
    return src
