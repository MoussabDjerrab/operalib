import numpy as np
from scipy.spatial.distance import pdist, cdist



def gaussparam(X,midkval=0.5):
    """ Decomposable gaussian kernel 
        B*k_gauss over the scalar gaussian
        estimates the gamma-parameter for a gaussian kernel
        defaults to 0.5
    """

    # if too large, choose randomly 2000 points
    n = len(X)
    if n > 2000:
        perm = np.random.permutation(n)
        j=0
        newX = [[]*len(X[0])]**2000
        while j < 2000 : 
            for i in perm : 
                newX[j][:] = X[i][:]
                j=j+1
        X = newX
    D = pdist(X)
    D = D**2
    gamma = - np.log(midkval) / np.mean(D);
    return gamma

def dcgauss(X1,X2,gamma,B):
    """ decomposable gaussian kernel B*k_gauss over the scalar gaussian """
    nrowX1 = X1.shape[0]
    nrowX2 = X2.shape[0]
    nrowB  = np.size(B[:,0])
    if X1.ndim<2 : X1 = np.array([X1]).T
    if X2.ndim<2 : X2 = np.array([X2]).T
    #TODO s = gaussparam(X1+X2)
    s1 = np.sqrt(gaussparam(X1))
    s2 = np.sqrt(gaussparam(X2))
    s = s1*s2
    K = np.tile(B,(nrowX1,nrowX2)) * np.exp(-gamma * s * np.kron(cdist(X1, X2,'euclidean')**2, np.ones((nrowB,nrowB))))
    return K