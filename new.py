from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pandas as pd
import numpy as np
import dill

seed = 13

def load_dill(fname):
    with open(fname, 'rb') as f:
        return dill.load(f)

def dump_dill(fname, obj):
    with open(fname, 'wb') as f:
        dill.dump(obj, f)
    return None

def get_weights(i, dists, classes, labels, n=5, eta=0.01):
    idx_n = dists[i].argsort()[1:n+1]
    weights = np.array([(labels[idx_n]==c).sum() for c in classes], dtype=np.float)
    weights += np.ones(classes.shape[0], dtype=np.float)*eta # Inject some noise!
    weights = weights/np.linalg.norm(weights, ord=1)
    return weights

def get_noise(X, y, classes, noise_size=0.1, max_iter=5):
    y_new = y.copy()
    idx = np.arange(y.shape[0])
    # precompute all distances
    dists = pairwise_distances(X, X)
    # add noise
    for n in range(max_iter):
        idx = np.random.permutation(idx)
        y_last = y_new.copy() # For methods that depends on noise prop
        for i in idx:
            weights = get_weights(i, dists, classes, y_last)
            y_new[i] = np.random.choice(classes, p=weights)
            frac = np.mean(y_new != y)
            if frac >= noise_size:
                break
        if frac >= noise_size:
            break
    return y_new
    
if __name__ == '__main__':

    # Generate data
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.25,0.75],
                               hypercube=True, class_sep=0.9, random_state=seed)

    classes = np.array([0,1])

    for noise_size in np.arange(0.01, 0.21, 0.01):
        print('noise_size=%0.2f' % noise_size)
        # Get NNAR noisy data
        y_nnar = []
        for i in tqdm(range(2000)):
            y_new = get_noise(X, y, classes, noise_size=noise_size)
            y_nnar.append(y_new)
        y_nnar = np.array(y_nnar)
        
        # Dump results to file
        data = {'X':X, 'y':y, 'y_nnar':y_nnar}
        dump_dill('nnar_blobs_%0.2f.dill' % (noise_size), data)