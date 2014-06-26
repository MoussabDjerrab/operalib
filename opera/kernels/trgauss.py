import numpy as np


def trgauss(X1,X2,gamma=1):
    nrowX1 = np.size(X1[:,0])
    nrowX2 = np.size(X2[:,0])
    ncolX  = np.size(X1[0,:])
    X1expand = np.tile(np.reshape(X1.T,((nrowX1*ncolX),1)),(1,(nrowX2*ncolX)))
    X2expand = np.tile(np.reshape(X2.T,(1,(nrowX2*ncolX))),((nrowX1*ncolX),1))
    K = np.exp(-gamma*(X1expand-X2expand)**2)
    return K