import numpy as np
from scipy.spatial.distance import cdist

def gaussianKernel(X,Y,gamma):
    (_,p) = X.shape
    return np.exp(-gamma * ((np.tile(X.T, (1,p)) - np.tile(Y,(p,1)))**2))

def gramMatrix(X1,X2,B,gamma):
    nrowX1 = np.size(X1[:,0])
    nrowX2 = np.size(X2[:,0])
    nrowB  = np.size(B[:,0])
    K = np.tile(B,(nrowX1,nrowX2)) * np.tile(gaussianKernel(X1,X2,gamma),(nrowB,nrowB))
    return K