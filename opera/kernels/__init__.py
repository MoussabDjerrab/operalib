from dcgauss import dcgauss
from dcpoly  import dcpoly
from dclin   import dclin
from dccust  import dccust
from trgauss import trgauss
from trpoly  import trpoly
from trlin   import trlin
from trcust  import trcust
from gramMatrix import gramMatrix
import numpy as np

def createB(y,B="identity"):
    """ Choose our B
    B is identity
        out is the identity matrix with y size
    B is covariance
        out is the covariance matrix of y
    B is a matrix
        out is B
    """
    if B == "identity" or B == None : 
        B = np.identity(len(y[0,:]))
    elif B == "cov" :
        B = np.cov(y)
    #elif B == "learn" : 
        #B = kernels.learn()
    return B
def chooseFunctionKernel(ovkernel="dc",kernel="gauss",c=1,d=3,gamma=1,B="identity"):
    """ Choose a kernel function
    n is the number of column of y
    ovkernel is : 
        dc : transformable : K(x,z)_(ij) = k(xi,zi)
        tf : decomposable : K(x,z) = B x k(x,z)
        custom : the user give a K
    kernel is : 
        gauss : gaussian kernel
        linear : linear kernel
        poly : polunomial kernel
        custom : the user give a k
    other parameter : 
        c and d for linear kernel (default c=1 and d=3) 
        g for gaussian kernel (default g=1)
        B for decomposable type (default B='identity')
            identity : B is id
            cov : B is cov(Y)
            learn : the algo learn B
            custom : the user give a B
    """

        
    if ovkernel=="dc" or ovkernel == None :
        if kernel == "gauss" or kernel == None :
            def f(X1,X2,y): return dcgauss(X1,X2,gamma,createB(y,B))
        elif kernel == "linear" : 
            def f(X1,X2,y): return dclin(X1,X2,createB(y,B))
        elif kernel == "polynomial" : 
            def f(X1,X2,y): return dclin(X1,X2,c,d,createB(y,B))
        elif kernel.__class__ == np.ndarray :
            def f(X1,X2,y): return kernel.copy
    elif ovkernel == "tr" : 
        if kernel == "gauss"  or kernel == None :
            def f(X1,X2,y): return trgauss(X1,X2,gamma)
        elif kernel == "linear" : 
            def f(X1,X2,y): return trlin(X1,X2)
        elif kernel == "polynomial" : 
            def f(X1,X2,y): return trpoly(X1,X2,c,d)
        elif kernel.__class__ == np.ndarray :
            def f(X1,X2,y): return kernel.copy()
    elif ovkernel.__class__ == np.ndarray : 
        def f(X1,X2,y): return ovkernel.copy()
    
    
    return f


