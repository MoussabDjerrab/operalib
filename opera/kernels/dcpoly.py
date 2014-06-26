import numpy as np

def __init__(X1,X2,c,d,B):
    """ decomposable polynomial kernel B*k_gauss over the scalar gaussian """
    nrowX1 = np.size(X1[:,0])
    nrowX2 = np.size(X2[:,0])
    ncolX  = np.size(X1[0,:])
    nrowB  = np.size(B[:,0])
    K = np.tile(B,(nrowX1,nrowX2)) * (np.kron(X1*X2.T/ncolX,np.ones((nrowB,nrowB)))+c)**d
    return K