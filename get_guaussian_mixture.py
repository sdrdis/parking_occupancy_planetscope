import tifffile
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.misc
import scipy.ndimage
from joblib import dump, load
from sklearn.mixture import GaussianMixture
import scipy.stats
import numpy as np
from sklearn.decomposition import PCA


def _estimate_template_pearsons(im1_np, im2_np, mask_np, size=2):
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
    
def estimate_template_pearsons(ims, median_np, mask_np):
    return apply_to_each(ims, median_np, mask_np, _estimate_template_pearsons)

def apply_to_each(ims, median_np, mask_np, feature_fn):
    out = []
    for i in range(ims.shape[0]):
        out.append(feature_fn(ims[i], median_np, mask_np))
    return np.array(out)

def get_gts():
    gts = {}
    with open('parking_data_v3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                filenames = row
            else:
                for i in range(len(row)):
                    if (row[i] == 'X'):
                        continue
                    gts[filenames[i][:-4]] = float(row[i])
            line_count += 1
    return gts
    

    
def _diff_feature(im_np, median_np, mask_np):
    return np.mean(im_np, 2)# - np.mean(median_np, 2)
    
def diff_feature(ims, median_np, mask_np):
    return apply_to_each(ims, median_np, mask_np, _diff_feature)
    
def _gradient_feature(im_np, median_np, mask_np):
    return np.max(scipy.ndimage.morphological_gradient(im_np, size=(3,3,1)), axis=2)# - np.max(scipy.ndimage.morphological_gradient(median_np, size=(3,3,1)), axis=2)
    
def gradient_feature(ims, median_np, mask_np):
    return apply_to_each(ims, median_np, mask_np, _gradient_feature)[:,:,:,np.newaxis]
    
def patch_pca_diff(ims, median_np, mask_np, nb_components=20, size=2):
    #print (ims.shape, median_np.shape)
    imsm = np.vstack((ims, median_np[np.newaxis,:,:,:]))
    
    X = []
    for i in range(imsm.shape[0]):
        im_np = imsm[i]
        for y in range(size, im_np.shape[0] - size):
            for x in range(size, im_np.shape[1] - size):
                if (mask_np[y,x] == False):
                    continue
                win1_np = im_np[y-size:y+size+1,x-size:x+size+1]
                X.append(win1_np.flatten())

    X = np.array(X)
    pca = PCA(n_components=nb_components)
    pca.fit(X)
    Xt = pca.transform(X)
    '''
    print(pca.explained_variance_ratio_)

    print(pca.singular_values_)
    '''
    Xi = 0
    out = []
    for i in range(imsm.shape[0]):
        im_np = imsm[i]
        out_np = np.zeros((im_np.shape[0], im_np.shape[1], nb_components))
        for y in range(size, im_np.shape[0] - size):
            for x in range(size, im_np.shape[1] - size):
                if (mask_np[y,x] == False):
                    continue
                out_np[y,x] = Xt[Xi]
                Xi += 1
        out.append(out_np)
    out = np.array(out)
    out_im = out[:-1]
    out_median = out[-1:]
    diffs = np.sum(np.abs(out_im - out_median), axis=3)
    return diffs
    print (out.shape, out_im.shape, out_median.shape, diffs.shape)
    sys.exit()
    
    
    return out
    print (out.shape)
    print(pca.explained_variance_ratio_)

    print(pca.singular_values_)
    sys.exit()

def compute_features(ims, median_np, mask_np, features=[diff_feature, gradient_feature, estimate_template_pearsons, patch_pca_diff]): # estimate_template_pearsons, diff_feature, gradient_feature, patch_pca_diff
    outs = []
    for feature in features:
        out_np = feature(ims, median_np, mask_np)
        if (len(out_np.shape) == 4):
            for i in range(out_np.shape[3]):
                outs.append(out_np[:,:,:,i])
        else:
            outs.append(out_np)
    outs = np.array(outs)
    outs = np.moveaxis(outs, 0, -1)
    
    return np.array(outs)
    
#folder = 
#path = 'planet_data/stabs'
path = 'planet_data/site_1_stabilized_f'
predictions_path = 'predictions'
gts = get_gts()
mask_np = scipy.misc.imread('mask.png')[:,:,0] > 127
mask_np = scipy.ndimage.binary_erosion(mask_np, iterations=7)
scipy.misc.imsave('ex_mask.png', mask_np.astype(float))



ims = []
ys = []
filenames = os.listdir(path)
filenames_f = []
for filename in filenames:
    fileid = filename[:-len('.tif')]
    if (fileid not in gts):
        continue
    filepath = join(path, filename)
    im_np = tifffile.imread(filepath)
    
    ims.append(im_np)
    filenames_f.append(filename)
    ys.append(gts[fileid])

ims = np.array(ims)
ys = np.array(ys)
median_np = np.median(ims, axis=0)

outs = compute_features(ims, median_np, mask_np)
outs = outs[:, mask_np, :]

all = outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[2]))
'''
print (ims.shape)
print (ys.shape)
print (median_np.shape)
print (outs.shape)
print (all.shape)
sys.exit()

#ims.append([s1_np, s2_np])
#ys.append(gts[fileid])
print (pos.shape)
'''

print ('NB COMPONENTS:', all.shape[1])

plt.clf()
plt.gca()

gmm = GaussianMixture(n_components=2).fit(all)
labels = gmm.predict(all)
probas = 1 - gmm.predict_proba(all)[:,0]

if (all.shape[1] == 2):
    plt.title('Gaussian mixture on all pixels (n=2)')
    plt.scatter(all[:, 0], all[:, 1], c=labels, s=20, cmap='viridis')
    plt.savefig('figures/gaussian_mixtures_all.png')

    dump(gmm, 'gmm.joblib')


#print (all[labels == 0], np.std(all[labels == 0], 0), np.mean(all[labels == 0], 0), all[labels == 1], np.std(all[labels == 1], 0), np.mean(all[labels == 1], 0))
#sys.exit()

#labels = gmm_small.predict(all)
zero_label = np.sum(np.std(all[labels == 0], 0)) <= np.sum(np.std(all[labels == 1], 0))

if (not zero_label):
    labels = 1-labels
    probas = 1-probas

labels = labels.reshape((ims.shape[0], np.sum(mask_np)))
probas = probas.reshape((ims.shape[0], np.sum(mask_np)))


'''
occupancies = []
for i in range(labels.shape[0]):
    sel_np = np.logical_or(probas[i] < 0.025, probas[i] > 0.975)
    occupancies.append(np.mean(labels[i, sel_np]))
occupancies = np.array(occupancies)
'''
occupancies = np.mean(labels, 1)


plt.clf()
plt.gca()
plt.scatter(occupancies, ys)
plt.xlabel('Occupancy rate estimated')
plt.ylabel('Occupancy rate ground truth')
plt.savefig('figures/prediction_mixture.png')
print (scipy.stats.pearsonr(occupancies, ys))

for i in range(labels.shape[0]):
    #pred_np = np.zeros(ims.shape[1:3])
    #pred_np[mask_np] = labels[i]
    tifffile.imsave(join(predictions_path, filenames_f[i][:-4] + '_1.tif'), ims[i])
    print (ims[i].shape, labels.shape)
    ims[i][mask_np] = labels[i,:,np.newaxis] * 65000
    tifffile.imsave(join(predictions_path, filenames_f[i][:-4] + '_2.tif'), ims[i])
    #scipy.misc.imsave(join(predictions_path, filenames_f[i][:-4] + '.png'), pred_np)
    

print ('CORR:', scipy.stats.pearsonr(occupancies, ys))