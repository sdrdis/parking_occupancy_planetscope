import tifffile
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.misc
import scipy.ndimage
from joblib import dump, load

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

path = 'planet_data/stabs'
gts = get_gts()
mask_np = scipy.misc.imread('mask.png')[:,:,0] > 127
mask_np = scipy.ndimage.binary_erosion(mask_np, iterations=7)
scipy.misc.imsave('ex_mask.png', mask_np.astype(float))
median_np = tifffile.imread('median.tif')
median_s1_np = np.mean(median_np, 2)
median_s2_np = scipy.ndimage.morphological_gradient(median_np, size=(3,3,1))
median_s2_np = np.max(median_s2_np, axis=2)
median_s_np = np.array([median_s1_np, median_s2_np])
ims = []
ys = []
for filename in os.listdir(path):
    fileid = filename[:-len('_band_RGB_stab.tif')]
    if (fileid not in gts):
        continue
    filepath = join(path, filename)
    im_np = tifffile.imread(filepath)
    
    s1_np = np.mean(im_np, 2)
    s2_np = scipy.ndimage.morphological_gradient(im_np, size=(3,3,1))
    s2_np = np.max(s2_np, axis=2)
    ims.append([s1_np, s2_np])
    ys.append(gts[fileid])
    
amb_threshold = 0.25
ims = np.array(ims)
threshold = 0.2
ys = np.array(ys)
print (ims.shape, mask_np.shape, ims[ys < 0.5].shape)
# 180:185, 186:191
xlims = [[6000, 10000], [0, 2000]]
xlabels = ['Pixel value', 'Gradient value']
for s in range(2):
    sel_ims = ims[:,s,:,:]
    neg = sel_ims[ys < threshold][:,180:185, 186:191].flatten()
    pos = sel_ims[ys > threshold][:,180:185, 186:191].flatten()
    plt.clf()
    plt.cla()
    plt.hist(neg, fc=(0, 0, 1, 0.5), bins=np.arange(0, 12000, 100), label='Low occupancy')
    plt.hist(pos, fc=(1, 0, 0, 0.5), bins=np.arange(0, 12000, 100), label='High occupancy')
    plt.xlim(xlims[s][0],xlims[s][1])
    plt.xlabel(xlabels[s])
    plt.ylabel('Number of points')
    plt.title('Histogram low vs high occupancy')
    plt.legend()
    plt.savefig('figures/1dplot_' + str(s) + '_hist.png')

    pxs = []
    pys = []
    total_points = 0
    total_amb_points = 0
    for i in np.arange(0, 12000, 100):
        nb_neg = np.sum(np.logical_and(neg >= i, neg < i + 100))
        nb_pos = np.sum(np.logical_and(pos >= i, pos < i + 100))
        nb_total = nb_neg + nb_pos
        if (nb_total == 0):
            continue
        pxs.append(i + 50)
        
        prob = nb_pos / nb_total
        total_points += nb_total
        if (not((prob < amb_threshold or prob > 1 - amb_threshold))):
            total_amb_points += nb_total
        
        
        pys.append(prob)
        
    print ('POINTS STATS:', total_amb_points / total_points, total_amb_points, total_points)
    plt.clf()
    plt.cla()
    plt.plot(pxs, pys)
    plt.xlabel(xlabels[s])
    plt.ylabel('Ratio')
    plt.title('Percentage of high occupancy cases')
    plt.xlim(xlims[s][0],xlims[s][1])
    plt.savefig('figures/1dplot_' + str(s) + '_ratio.png')


width = 12000
height = 12000
step = 100
total_points = 0
total_amb_points = 0
pos_np = np.zeros((int(height / step), int(width / step)))
neg_np = np.zeros((int(height / step), int(width / step)))
med_np = np.zeros((int(height / step), int(width / step)))
ratio_np = np.zeros((int(height / step), int(width / step)))
ratio_np[:] = np.nan
neg = ims[ys < threshold][:,:,180:185, 186:191]
pos = ims[ys > threshold][:,:,180:185, 186:191]
med = median_s_np[:,180:185, 186:191]
for x in np.arange(0, width, step):
    for y in np.arange(0, height, step):
        #print ((neg[:,0] >= x).shape)
        
        #print ((ims[ys < threshold][:,0,180:185, 186:191] >= x).shape)
        #sys.exit()
        nb_neg = np.sum(np.logical_and(np.logical_and(neg[:,0] >= x, neg[:,0] < x + step), np.logical_and(neg[:,1] >= y, neg[:,1] < y + 100)))
        nb_pos = np.sum(np.logical_and(np.logical_and(pos[:,0] >= x, pos[:,0] < x + step), np.logical_and(pos[:,1] >= y, pos[:,1] < y + 100)))
        nb_med = np.sum(np.logical_and(np.logical_and(med[0] >= x, med[0] < x + step), np.logical_and(med[1] >= y, med[1] < y + 100)))
        
        nb_total = nb_neg + nb_pos
        
        
        pos_np[int(y/step), int(x/step)] = nb_pos
        neg_np[int(y/step), int(x/step)] = nb_neg
        med_np[int(y/step), int(x/step)] = nb_med
        
        if (nb_total == 0):
            continue
        prob = nb_pos / nb_total
        total_points += nb_total
        if (not((prob < amb_threshold or prob > 1 - amb_threshold))):
            total_amb_points += nb_total
        ratio_np[int(y/step), int(x/step)] = prob

plt.imshow(med_np)
plt.title('Median image')
plt.xlabel('Pixel value (x 100)')
plt.ylabel('Gradient value (x 100)')
plt.xlim(65,95)
plt.ylim(0,25)
plt.savefig('figures/2dplot_median.png')

plt.imshow(neg_np)
plt.title('Low occupancy images')
plt.xlim(65,95)
plt.ylim(0,25)
plt.savefig('figures/2dplot_neg.png')



plt.imshow(pos_np)
plt.title('High occupancy images')
plt.xlim(65,95)
plt.ylim(0,25)
plt.savefig('figures/2dplot_pos.png')

print ('POINTS STATS:', total_amb_points / total_points, total_amb_points, total_points)
plt.imshow(ratio_np)
plt.title('Occupancy probability')
plt.xlim(65,95)
plt.ylim(0,25)
plt.savefig('figures/2dplot_ratio.png')

def reshape_list(lst):
    lst = np.moveaxis(lst, 1, 3)
    lst = lst.reshape((lst.shape[0] * lst.shape[1] * lst.shape[2], lst.shape[3]))
    return lst

from sklearn.mixture import GaussianMixture
#ims.append([s1_np, s2_np])
#ys.append(gts[fileid])
print (pos.shape)
pos = pos - med
neg = neg - med
pos = reshape_list(pos)
neg = reshape_list(neg)
all = np.vstack((pos, neg))

plt.clf()
plt.gca()

gmm_small = GaussianMixture(n_components=2).fit(all)
labels_small = gmm_small.predict(all)
plt.title('Gaussian mixture (n=2)')
plt.scatter(all[:, 0], all[:, 1], c=labels_small, s=20, cmap='viridis')
plt.savefig('figures/gaussian_mixtures.png')



all = ims - median_s_np
all = all[:,:,mask_np]
all = np.moveaxis(all, 1, 2)
all = all.reshape((all.shape[0] * all.shape[1], all.shape[2]))

plt.clf()
plt.gca()

gmm = GaussianMixture(n_components=2).fit(all)
labels = gmm.predict(all)
plt.title('Gaussian mixture on all pixels (n=2)')
plt.scatter(all[:, 0], all[:, 1], c=labels, s=20, cmap='viridis')
plt.savefig('figures/gaussian_mixtures_all.png')

dump(gmm, 'gmm.joblib')



#labels = gmm_small.predict(all)
zero_label = np.std(all[labels == 0]) <= np.std(all[labels == 1])

if (not zero_label):
    labels = 1-labels

labels = labels.reshape((ims.shape[0], np.sum(mask_np)))
occupancies = np.mean(labels, 1)

plt.clf()
plt.gca()
plt.scatter(occupancies, ys)
plt.xlabel('Metric')
plt.ylabel('Occupancy rate ground truth')
plt.savefig('figures/prediction_mixture.png')
print (scipy.stats.pearsonr(occupancies, ys))