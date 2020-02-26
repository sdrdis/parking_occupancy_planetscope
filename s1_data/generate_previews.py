import tifffile
import os
from os.path import join, isfile
import scipy.misc
import numpy as np

path = 'site_1_stabilized_f'
to_path = 'stabs_previews'
extracts_path = 'D:\Developments\parking_detection\extracts_s1'
for filename in os.listdir(path):
    filepath = join(path, filename)
    im_np = tifffile.imread(filepath)
    im_np = im_np[:,:,[1,1,1]]
    m = np.mean(im_np)
    std = np.std(im_np)
    im_np = (im_np - 0.06) / 0.12 # -0.02 / 0.04
    im_np[im_np < 0] = 0
    im_np[im_np > 1] = 1
    im_np *= 255
    im_np = im_np.astype('uint8')

    print (filename)
    ex_path = join(extracts_path, filename[:-len('.png')] + '.png')
    if (not (isfile(ex_path))):
        continue
    ex_np = scipy.misc.imread(ex_path)
    
    ex_np = scipy.misc.imresize(ex_np,(362, 641))
    
    imf_np = np.zeros((362, 641+362, 3), dtype='uint8')
    imf_np[:,:641] = ex_np
    imf_np[:,642:] = im_np
    scipy.misc.imsave(join(to_path, filename[:-4] + '.png'), imf_np)