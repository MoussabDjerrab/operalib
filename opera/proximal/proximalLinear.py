import numpy as np
from proximalGeneral import __init__ as proximalGeneral
from scipy.sparse import linalg as LAs
import numpy.linalg as LA

"""
AUXILIAR FUNCTION
"""
def spectralradius(M):
    l = LAs.eigsh(M, 1,maxiter=100, return_eigenvectors=False)[0]
    return l
def normvector(x,norm="l1"):
    if norm.upper() == "L1" : 
        return LA.norm(x,1)
    elif norm.upper() == "L2" : 
        return LA.norm(x,2)
    elif norm.upper() == "mixed" : 
        return np.mean((LA.norm(x,1),LA.norm(x,2)))
     

def __init__(K, y, init=None, mu=1, muX=1, norm='l1', maxiters=100, n=1, eps=1.e-3):
    """ABSTRACT : Learning x with a norm constraint on the coefficients
    REFERENCE : Beck and Teboulle (2010) Gradient-based algorithms with applications to signal-recovery problems
    INPUTS :
    K        : ([N*p,N*p]) Gram matrix
    Yvec     : ([N*p,1]) output vector
    niters    : number of iterations
    C0       : ([N*p,1]) initial c-vector
    muH  : (positive) hyperparameter for squared norm of h
    muC  : (positive) hyperparameter for the L1-norm constraint
    OUTPUTS : 
    C  : ([N*p,1]) solution
    AUTHORS  : Nehemy LIM, AROBAS group, IBISC, Universite d'Evry, France
               Markus Heinonen 
               Tristan TCHILINGUIRIAN 
    """
    L = 2* spectralradius(np.dot(K,K))
    if init is None :
        X = np.linalg.solve(K + mu*np.identity(K.shape[0]), y)
    else :
        X = init.copy()
    def gradient(x) : 
        return np.dot(K,np.dot(K+mu*np.identity(K.shape[0]),x)-y)       
    def objective(x):
        LA.norm(np.dot(K,x)-y)**2 + normvector(x)
    return proximalGeneral(gradient,L,init=X,maxiters=maxiters,norm=norm,mu=mu,n=n,eps=eps)
    