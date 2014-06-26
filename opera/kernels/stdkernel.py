import numpy as np
from scipy.spatial.distance import pdist, squareform

def gaussiankernel(X,gamma):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = np.exp(-gamma * pairwise_dists ** 2)
    return K
def polynomialkernel(X,c,d):
    dot_product = np.dot(X,X.T)
    K = (dot_product + c)**d
    return K
def linearkernel(X):
    return polynomialkernel(X, 0, 1)