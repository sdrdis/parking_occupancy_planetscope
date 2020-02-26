import tifffile
import os
from os.path import join, isfile
import scipy.misc
import numpy as np


path = 'site_1_stabilized_f'
to_path = 'stabs_previews'
extracts_path = 'D:\Developments\parking_detection\extracts'
for filename in os.listdir(path):
    filepath = join(path, filename)
    im_np = tifffile.imread(filepath)
    m = np.mean(im_np)
    std = np.std(im_np)
    im_np = (im_np - m) / std
    im_np /= 1.5
    im_np[im_np < -1] = -1
    im_np[im_np > 1] = 1
    im_np = (im_np + 1) / 2
    im_np *= 255
    im_np = im_np.astype('uint8')

    print (filename)
    ex_path = join(extracts_path, filename[:-len('.png')] + '.png')
    if (not (isfile(ex_path))):
        continue
    ex_np = scipy.misc.imread(ex_path)
    
    ex_np = scipy.misc.imresize(ex_np,(363, 642))
    
    imf_np = np.zeros((363, 642+362, 3), dtype='uint8')
    imf_np[:,:642] = ex_np
    imf_np[:,642:] = im_np
    scipy.misc.imsave(join(to_path, filename[:-4] + '.png'), imf_np)