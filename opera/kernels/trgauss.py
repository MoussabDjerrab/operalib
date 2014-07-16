import numpy as np


def trgauss(X1,X2,gamma=1):
    """transformable gaussian kernel 
        with K(x,x')_ij = k_gauss(x(i),x'(j))
    assumes dims(x) = dims(y)
    """
    nrowX1 = X1.shape[0]
    nrowX2 = X2.shape[0]
    if X1.ndim < 2 : 
        ncolX = 1
    else :
        ncolX  = X1.shape[1]
    X1expand = np.tile(np.reshape(X1.T,((nrowX1*ncolX),1)),(1,(nrowX2*ncolX)))
    X2expand = np.tile(np.reshape(X2.T,(1,(nrowX2*ncolX))),((nrowX1*ncolX),1))
    K = np.exp(-gamma*(X1expand-X2expand)**2)
    return K