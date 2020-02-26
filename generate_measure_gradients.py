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
import scipy.stats

def estimate_template_matching_diff(im1_np, im2_np, mask_np, size=2):
    diff_np = np.ones(im1_np.shape[:2])
    for y in range(size, im1_np.shape[0] - size):
        for x in range(size, im2_np.shape[1] - size):
            if (mask_np[y,x] == False):
                continue
            win1_np = im1_np[y-size:y+size+1,x-size:x+size+1]
            win2_np = im2_np[y-size:y+size+1,x-size:x+size+1]
            sel_np = mask_np[y-size:y+size+1,x-size:x+size+1,np.newaxis]
            score = np.sum(win1_np * win2_np * sel_np) / np.sqrt(np.sum((win1_np * sel_np) ** 2) * np.sum((win2_np * sel_np) ** 2))

            diff_np[y, x] = score
    diff_np[np.logical_not(mask_np)] = 1
    diff_np = 1 - diff_np
    return diff_np
    
def estimate_template_pearsons(im1_np, im2_np, mask_np, size=2):
    diff_np = np.ones(im1_np.shape[:2])
    for y in range(size, im1_np.shape[0] - size):
        for x in range(size, im2_np.shape[1] - size):
            if (mask_np[y,x] == False):
                continue
            win1_np = im1_np[y-size:y+size+1,x-size:x+size+1]
            win2_np = im2_np[y-size:y+size+1,x-size:x+size+1]
            sel_np = mask_np[y-size:y+size+1,x-size:x+size+1]
            #score = np.sum(win1_np * win2_np * sel_np) / np.sqrt(np.sum((win1_np * sel_np) ** 2) * np.sum((win2_np * sel_np) ** 2))

            
            win1_np = win1_np[sel_np].flatten()
            win2_np = win2_np[sel_np].flatten()
            score = scipy.stats.pearsonr(win1_np, win2_np)[0]
            

            diff_np[y, x] = score
    diff_np[np.logical_not(mask_np)] = 1.0
    diff_np = 1 - diff_np
    return diff_np


def symmetry_score(kernel_np):
    sum_up = np.sum(kernel_np[:int(kernel_np.shape[0]/2),:])
    sum_down = np.sum(kernel_np[kernel_np.shape[0]-int(kernel_np.shape[0]/2):,:])
   
    sum_left = np.sum(kernel_np[:,:int(kernel_np.shape[0]/2)])
    sum_right = np.sum(kernel_np[:,kernel_np.shape[0]-int(kernel_np.shape[0]/2):])
   
    score_up_down = sum_up - sum_down
    score_left_right = sum_left - sum_right
   
    return abs(score_up_down) + abs(score_left_right)
   
def shift_kernel(kernel_np, pos):
    new_kernel_np = scipy.ndimage.shift(kernel_np, (pos[0], pos[1]))
    new_kernel_np /= np.sum(new_kernel_np)
    return new_kernel_np
   
   
def opt_kernel(opt_params, *params):
    y, x = opt_params
    kernel_np, = params
    return symmetry_score(shift_kernel(kernel_np, (y,x)))

def get_best_kernel(kernel_np, mult=1.0):
    rranges = (slice(-3*mult, 3.01*mult, mult/2.0), slice(-3*mult, 3.01*mult, mult/2.0))
    (y, x) = scipy.optimize.brute(opt_kernel, rranges, args=(kernel_np,))
    #print (y,x)
    return shift_kernel(kernel_np, (y,x))
    

def get_kernel_stability(kernel_np, mult=1):
    radius = kernel_np.shape[0] / 2.0
    vals = 0
    weights = 0
    for i in range(1, int(radius - 1), mult):
        rr, cc = skimage.draw.circle_perimeter(int(radius), int(radius), i)
        nb = np.sum(kernel_np[rr,cc])
        vals += nb * np.std(kernel_np[rr,cc])
        weights += nb
    return vals / weights
    
def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
    
def gini_blur(x):
    radius = int(x.shape[0] / 2)
    rr, cc = skimage.draw.circle(radius, radius, radius)
    return gini(np.abs(x[rr, cc]))

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
metric = 'gradient'
metric = 'diff_median'
#metric = 'correlation_median'
metric = 'correlation_bounded_median'
#metric = 'precomputed:nn/diff/preds_vals'
metric = 'correlation_pearsons_median'

list_filenames = []
for filename in os.listdir(stabs_path):
    fileid = filename[:-len('_band_RGB_stab.tif')]
    if (fileid not in gts):
        continue
    list_filenames.append(filename)

blurs = []
y = []
for filename in list_filenames:
    fileid = filename[:-len('_band_RGB_stab.tif')]
    kernel_np = tifffile.imread(join(kernels_path, fileid + '.tif'))
    #kernel_np = scipy.ndimage.zoom(kernel_np, 8.0)
    #kernel_np = get_best_kernel(kernel_np, 8.0)
    #blur_estimation = get_kernel_stability(kernel_np, 8)
    kernel_np = get_best_kernel(kernel_np, 1.0)
    blur_estimation = gini_blur(kernel_np) #np.linalg.norm(kernel_np, 2)
    blurs.append(blur_estimation)
    y.append(gts[fileid])

ims_np = []
for filename in list_filenames:
    im_np = tifffile.imread(join(stabs_path, filename))
    ims_np.append(im_np)
ims_np = np.array(ims_np)
median_np = np.median(ims_np, 0)

bounds = []
for i in range(median_np.shape[2]):
    bounds.append([median_np[mask_np,i].min() - 1000, median_np[mask_np,i].max() + 1000])

tifffile.imsave('median.tif', median_np)

bounded_median_np = apply_bounds(median_np, bounds)

x = []
for i in range(len(list_filenames)):
    filename = list_filenames[i]
    fileid = filename[:-len('_band_RGB_stab.tif')]
    im_np = ims_np[i] #tifffile.imread(join(stabs_path, filename))
    
    if (metric == 'gradient'):
        s_np = scipy.ndimage.morphological_gradient(im_np, size=(3,3,1))
        s_np = np.max(s_np, axis=2)
        val = np.mean(s_np[mask_np] > 500)
    
    if (metric == 'diff_median'):
        diff_np = im_np - median_np #np.max(np.abs(im_np - median_np), axis=2)
        #diff_np = scipy.ndimage.grey_closing(diff_np, size=(3,3))
        diff_np[np.logical_not(mask_np)] = 0
        # (141, 138, 234, 206)
        bbox = [0,0,0,0]
        bbox[0] = 139
        bbox[2] = 235
        bbox[1] = 124
        bbox[3] = 220
        tifffile.imsave('tmp/'+fileid+'_X.tif', im_np[bbox[0]:bbox[2], bbox[1]:bbox[3]])
        np.savez_compressed('tmp/'+fileid+'_y.npz',y=gts[fileid])
        val = np.mean(diff_np[mask_np] > 700)
        
    if (metric == 'diff_median_gradient'):
        s_np = scipy.ndimage.morphological_gradient(im_np, size=(3,3,1))
        s_np = np.max(s_np, axis=2)
        diff_np = np.max(np.abs(im_np - median_np), axis=2)
        tifffile.imsave('tmp/'+fileid+'.tif', diff_np)
        val = np.mean(np.logical_or(diff_np[mask_np] > 650, s_np[mask_np] > 550))
        
    if (metric == 'correlation_median'):
        s_np = estimate_template_matching_diff(im_np, median_np, mask_np, size=2)
        tifffile.imsave('tmp/'+fileid+'.tif', s_np)
        val = np.mean(s_np[mask_np] > 0.0003)
        
    if (metric == 'correlation_pearsons_median'):
        s_np = estimate_template_pearsons(im_np, median_np, mask_np, size=2)
        tifffile.imsave('tmp/'+fileid+'.tif', s_np)
        val = np.mean(s_np[mask_np] > 0.03)
        
    if (metric == 'correlation_bounded_median'):
        s_np = estimate_template_matching_diff(apply_bounds(im_np, bounds), bounded_median_np, mask_np, size=2)
        #s_np = scipy.ndimage.grey_closing(s_np, size=(3,3))
        tifffile.imsave('tmp/'+fileid+'.tif', s_np)
        val = np.mean(s_np[mask_np] > 0.005)
    
    if (metric.startswith('precomputed')):
        path = metric.split(':')[1]
        val = np.load(join(path, fileid + '.npz'))['prediction']
    
    x.append(val)
    print (filename, val, gts[fileid], blur_estimation)
 
idx = np.argsort(x)
for i in idx:
    filename = list_filenames[i]
    fileid = filename[:-len('_band_RGB_stab.tif')]
    print (x[i], gts[fileid], fileid)

print ('CORRELATION:', scipy.stats.pearsonr(x, y))
plt.scatter(x, y, c=blurs)
plt.ylabel('Occupancy rate ground truth')
plt.xlabel('Metric')
plt.show()

