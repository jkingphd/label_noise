from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from itertools import repeat
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

def get_noise(X, y, classes, noise_size=0.1, max_iter=50):
    eta = 0.01
    y_new = y.copy()
    for i in range(max_iter):
        knn = KNeighborsClassifier(n_neighbors=10).fit(X, y_new)
        weights = normalize(knn.predict_proba(X) + eta, norm='l1')
        y_new = np.array([np.random.choice(classes, p=w) for w in weights])
        for j, val in enumerate(y_new):
            y_test = np.hstack([y_new[:j], y[j:]])
            frac = np.mean(y_test != y)
            if frac >= noise_size:
                y_new = y_test.copy()
                break
        if frac >= noise_size:
            break
    return y_new
    
def dummy(tup):
    return get_noise(*tup)
    
if __name__ == '__main__':

    # Generate data
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1, weights=[0.25,0.20,0.55],
                               hypercube=True, class_sep=0.8, random_state=seed)
    
    classes = np.array([0,1,2])
    n = 2000
    
    for noise_size in np.arange(0.01, 0.21, 0.01):
        print('noise_size=%0.2f' % noise_size)
        # Get NNAR noisy data
        input = repeat((X, y, classes, noise_size, 50), n)
        with Pool(16) as pool:
            data = list(tqdm(pool.imap_unordered(dummy, input, chunksize=100), total=n))
        data = np.array(data)
        
        # Dump results to file
        output = {'X':X, 'y':y, 'y_new':data}
        dump_dill('tri_blobs_nnar_%0.2f.dill' % (noise_size), output)