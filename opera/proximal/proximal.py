import numpy as np
import numpy.linalg as LA

def proximal(gradient,norm='L1',mu=1,L=1,n=1):
    if norm.upper() == 'L1' : 
        return proxlasso(gradient,mu,L)
    elif norm.lower() == 'mixed' : 
        return proxlasso(gradient,mu,L,n)
    return



def proxl1(grad,mu,L):
    tmp2 = np.abs(grad)-(mu/L)
    # test = (tmp2>=0) but it is not working, i don't know why
    test = tmp2.copy()
    test[tmp2>=0] = 1
    test[tmp2<0] = 0
    Sol = test*tmp2*np.sign(grad)
    return Sol
def proxlasso(grad,mu=1,L=1,n=1):
    p = len(grad)/n
    Sol = np.zeros(len(grad))
    for i in range(n) : 
        rangeinf = (i-1)*p
        rangesup = (i)*p+1 # apply thresholding over slices
        tmp2 = 1 - mu/L * LA.norm( grad[rangeinf:rangesup] )
        if tmp2 >=0 : 
            tmp2 = 1
        else :
            tmp2 = 0
        Sol[rangeinf:rangesup] = tmp2 * grad[rangeinf:rangesup]
    return Sol