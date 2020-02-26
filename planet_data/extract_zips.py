from zipfile import ZipFile
from os.path import join, isdir
import os
import shutil
import scipy.ndimage
import tifffile
import numpy as np

from_path = 'walmart_1/zips'
to_path = 'walmart_1/ims'
tmp_path = 'tmp'
for filename in os.listdir(from_path):
    print (filename)
    with ZipFile(join(from_path, filename), 'r') as zipObj:
        if (isdir(tmp_path)):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
        # Extract all the contents of zip file in different directory
        zipObj.extractall(tmp_path)
        
        for f in os.listdir(tmp_path):
            if (f.endswith('_Analytic_clip.tif')):
                dn_f = f[:-9] + '_DN_udm_clip.tif'
                clouds_np = tifffile.imread(join(tmp_path, dn_f)) > 0
                clouds_np = scipy.ndimage.grey_opening(clouds_np, (15, 15))
                
                if (np.max(clouds_np) == False):
                    shutil.move(join(tmp_path, f), join(to_path, f))
        #sys.exit()