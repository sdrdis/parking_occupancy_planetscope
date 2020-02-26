import os
from os.path import join
import tifffile
import numpy as np
import scipy.ndimage
import csv
import matplotlib.pyplot as plt
import scipy.stats
import cv2
import scipy.optimize
import scipy.ndimage
import skimage.draw
import skimage.measure
from scipy import optimize

def get_gts():
    gts = {}
    with open('parking_data_v2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                filenames = row
            else:
                for i in range(len(row)):
                    gts[filenames[i][:-4]] = float(row[i])
            line_count += 1
    return gts
    
def apply_bounds(_im_np, bounds):
    im_np = _im_np.copy().astype(float)
    for i in range(len(bounds)):
        im_np[:,:,i] -= bounds[i][0]
        im_np[:,:,i] /= bounds[i][1] - bounds[i][0]
    im_np[im_np < 0] = 0
    im_np[im_np > 1] = 1
    return im_np
    
gts = get_gts()
stabs_path = 'planet_data/stabs'
kernels_path = 'planet_data/kernels'
mask_np = scipy.misc.imread('mask.png')[:,:,0] > 127
props = skimage.measure.regionprops(mask_np.astype(int))
bbox = props[0].bbox

list_filenames = []
for filename in os.listdir(stabs_path):
    fileid = filename[:-len('_band_RGB_stab.tif')]
    if (fileid not in gts):
        continue
    list_filenames.append(filename)


ims_np = []
ys = []
for filename in list_filenames:
    fileid = filename[:-len('_band_RGB_stab.tif')]
    im_np = tifffile.imread(join(stabs_path, filename))
    ims_np.append(im_np)
    ys.append(gts[fileid])
ys = np.array(ys)
ims_np = np.array(ims_np)
median_np = np.median(ims_np, 0)

median_s1_np = np.mean(im_np, 2)
median_s2_np = scipy.ndimage.morphological_gradient(median_np, size=(3,3,1))
median_s2_np = np.max(median_s2_np, axis=2)

xs = []
for i in range(len(list_filenames)):
    filename = list_filenames[i]
    fileid = filename[:-len('_band_RGB_stab.tif')]
    im_np = ims_np[i] #tifffile.imread(join(stabs_path, filename))
    
    s1_np = np.mean(im_np, 2)
    s1_np -= median_s1_np
    s2_np = scipy.ndimage.morphological_gradient(im_np, size=(3,3,1))
    s2_np = np.max(s2_np, axis=2)
    s2_np -= median_s2_np
    x = np.array([s1_np[mask_np], s2_np[mask_np]]).transpose()
    xs.append(x)
    
xs = np.array(xs)

print (xs.shape)
print (ys.shape)

def get_occupancies(xs, ys, s1, s2):
    return np.mean(np.logical_or(xs[:,:,0] < s1, xs[:,:,1] > s2), axis=1)

def occupancy(z, *params):
    s1, s2 = z
    xs, ys = params
    occupancies = get_occupancies(xs, ys, s1, s2)
    
    #score = np.mean(np.abs(occupancies - ys))
    score = 1 - scipy.stats.pearsonr(occupancies, ys)[0]
    print (s1, s2, score)
    return score

rranges = (slice(-1000, 0, 50), slice(0, 1000, 50))

resbrute = optimize.brute(occupancy, rranges, args=(xs,ys), full_output=True, finish=optimize.fmin)
occupancies = get_occupancies(xs, ys, resbrute[0][0], resbrute[0][1])
plt.scatter(occupancies, ys)
plt.xlabel('Metric')
plt.ylabel('Occupancy rate ground truth')
plt.show()
print (scipy.stats.pearsonr(occupancies, ys))