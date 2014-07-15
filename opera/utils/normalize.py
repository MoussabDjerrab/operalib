import numpy as np

def normalize(K,p=1):
    n = K.shape[0]/p
    Kdiag = np.zeros((p,n*p))
    for i in range(n) : 
        rmin = (i-1)*p+1
        rmax = i*p+1
        Kdiag[:,rmin:rmax]=K[rmin:rmax,rmin:rmax]
    return K/np.sqrt(np.dot(Kdiag.T,Kdiag))