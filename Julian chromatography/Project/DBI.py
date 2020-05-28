import numpy as np
from scipy.spatial.distance import euclidean, cdist, pdist, squareform

def db_index(X, y):
    """
    Davies-Bouldin index is an internal evaluation method for
    clustering algorithms. Lower values indicate tighter clusters that 
    are better separated.

    Arguments
    ----------
    X : 2D array (n_samples, embed_dim)
    Vector for each example.

    y : 1D array (n_samples,) or 2D binary array (n_samples, n_classes)
    True labels for each example.

    Returns
    ----------
    dbi : float
        Calculated Davies-Bouldin index.
    """
    # get unique labels
    if y.ndim == 2:
        y = np.argmax(axis=1)
    uniqlbls = np.unique(y)
    n = len(uniqlbls)
    # pre-calculate centroid and sigma
    centroid_arr = np.empty((n, X.shape[1]))
    sigma_arr = np.empty(n)
    for i,k in enumerate(uniqlbls):
        Xk = X[np.where(y==k)[0],...]
        Ak = np.mean(Xk, axis=0)
        centroid_arr[i,...] = Ak
        sigma_arr[i,...] = np.mean(cdist(Xk, Ak.reshape(1,-1)))
    # loop over non-duplicate cluster pairs
    dbi = 0
    for i in range(n):
        max_Rij = 0
        for j in range(n):
            if j != i:
                Rij = np.divide(sigma_arr[i] + sigma_arr[j], 
                                euclidean(centroid_arr[i,...], centroid_arr[j,...]))
                if Rij > max_Rij:
                    max_Rij = Rij
        dbi += max_Rij
    return dbi/n