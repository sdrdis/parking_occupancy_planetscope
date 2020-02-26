import os
from os.path import join
import numpy as np
import scipy.misc

extracts_path = 'extracts'
annotated_extracts_path = 'annotated_extracts'
mask_np = scipy.misc.imread('mask_over_images.png')
sel_np = mask_np[:,:,3] > 0
mask_np = mask_np[:,:,:3]
s = []
for filename in os.listdir(extracts_path):
    s.append(filename)
    continue
    im_np = scipy.misc.imread(join(extracts_path, filename))
    im_np[sel_np] = mask_np[sel_np]
    scipy.misc.imsave(join(annotated_extracts_path, filename), im_np)
    
with open('columns.txt', 'w') as f:
    f.write("\t".join(s))