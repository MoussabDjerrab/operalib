import numpy as np
from proximalGeneral import proximalGeneral
from opera.utils import spectralradius
import numpy.linalg as LA


def proximalLinear(K, y, init=None, mu=1, norm='l1', muX_1=1, muX_2=1, partitionX=None, partitionX_weight=None, maxiters=100, eps=None):
    """
    ABSTRACT : Learning x with a norm constraint on the coefficients
    REFERENCE : Beck and Teboulle (2010) Gradient-based algorithms with applications to signal-recovery problems
    INPUTS :
        K    : ([N*p,N*p]) Gram matrix
        y    : ([N*p,1]) output vector
        init    : ([N*p,1]) initial c-vector, if it is None then (K+I)*init = y
        mu  : (positive) hyperparameter for squared norm of h
        muX  : (positive) hyperparameter for the norm constraint
        norm : norm constraint
        maxiters : number of iterations
    OUTPUTS : 
        X  : ([N*p,1]) solution
    """
    L = 2* spectralradius(np.dot(K.T,K))
    if init is None :
        X = np.linalg.solve(K + mu*np.identity(K.shape[0]), y)
    else :
        X = init.copy()
    def gradient(x) : 
        return np.dot(K,np.dot(K+mu*np.identity(K.shape[0]),x)-y)       
    def objective(x):
        return LA.norm(np.dot(K,x)-y,2)**2 + mu/L*LA.norm(x,1)
        
    return proximalGeneral(L,X,gradient,objective,False,maxiters,norm,muX_1,muX_2,partitionX,partitionX_weight,eps,debug=False)
    
