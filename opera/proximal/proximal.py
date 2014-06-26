import numpy as np
import numpy.linalg as LA

def __init__(gradient,norm='L1',mu=1,L=1,n=1):
    p = len(gradient)/n
    Sol = np.zeros(len(gradient))
    if norm.upper() == 'L1' : 
        tmp2 = np.abs(gradient)-(mu/L)
        # test = (tmp2>=0) but it is not working, i don't know why
        test = tmp2.copy()
        test[tmp2>=0] = 1
        test[tmp2<0] = 0
        Sol = test*tmp2*np.sign(gradient)
    elif norm.lower() == 'mixed' : 
        for i in range(n) : 
            rangeinf = (i-1)*p
            rangesup = (i)*p+1 # apply thresholding over slices
            tmp2 = 1 - mu/L * LA.norm( gradient[rangeinf:rangesup] )
            if tmp2 >=0 : 
                tmp2 = 1
            else :
                tmp2 = 0
            Sol[rangeinf:rangesup] = tmp2 * gradient[rangeinf:rangesup]
    return Sol