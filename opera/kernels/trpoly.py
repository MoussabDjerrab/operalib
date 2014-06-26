import numpy as np

def trpoly(X1,X2,c=0,d=1):
    nrowX1 = X1.shape[0]
    nrowX2 = X2.shape[0]
    ncolX  = X1.shape[1]
    X1expand = np.tile(np.reshape(X1.T,((nrowX1*ncolX),1)),(1,(nrowX2*ncolX)))
    X2expand = np.tile(np.reshape(X2.T,(1,(nrowX2*ncolX))),((nrowX1*ncolX),1))
    K = (X1expand*X2expand + c )**d
    return K