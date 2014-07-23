import numpy as np
from opera.kernels.dcgauss import dcgauss
from opera.kernels.trgauss import trgauss

def gaussianKernel(X,Y,gamma):
    p = X.shape[0]
    if X.ndim == 1 :
        X = np.array([X])
    if Y.ndim == 1 :
        Y = np.array([Y])
    return np.exp(-gamma * ((np.tile(X.T, (1,p)) - np.tile(Y,(p,1)))**2))

def gramMatrix(X1,X2,B,gamma,gamma2=0):
    n = X1.shape[0]
    T = X2.shape[0]
    p = B.shape[0]
    K = np.zeros((T*p,n*p))
    for t in range(T) :
        for i in range(n) :
            K[np.ix_(t*p+np.array(range(p)),i*p+np.array(range(p)))] = B * gaussianKernel((X1[i,:]).T, (X2[t,:]).T,gamma)
    return K

def gramMatrix_(X1,X2,B,gammadc,gammatr):
    #TODO 
    K = dcgauss(X1,X2,gammadc,B) * trgauss(X1,X2,gammatr)
    return K