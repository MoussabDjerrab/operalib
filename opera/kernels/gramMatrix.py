import numpy as np
from opera.kernels.dcgauss import dcgauss
from opera.kernels.trgauss import trgauss

def gaussianKernel(X,Y,gamma):
    (_,p) = X.shape
    return np.exp(-gamma * ((np.tile(X.T, (1,p)) - np.tile(Y,(p,1)))**2))

def gramMatrix(X1,X2,B,gammadc,gammatr):
    #TODO 
    K = dcgauss(X1,X2,B,gammadc) * trgauss(X1,X2,gammatr)
    return K