import numpy as np
from proximal import proximal
import numpy.linalg as LA


def proximalGeneral(gradient,L,init=None,objective=None,maxiters=100,norm='L1',mu=1,n=1,eps=1.e-3):
    """
    ABSTRACT : Learning gradient with a norm constraint on the coefficients
    REFERENCE : Beck and Teboulle (2010) Gradient-based algorithms with applications to signal-recovery problems
    INPUTS :
        gradient : 
        L : 
        init    : ([N*p,1]) initial c-vector, not None
        mu  : (positive) hyperparameter for squared norm of h
        muX  : (positive) hyperparameter for the norm constraint
        norm : norm constraint
        maxiters : number of iterations  
    OUTPUTS : 
        X  : ([N*p,1]) solution
    """

    if init is None : 
        print "proximalGeneral algorithm need an initial vector"
        return None
    
    tk = 1
    tkOld = 1
    zk = init.copy()
    zkOld = zk.copy()
    xk = init.copy()
    xkOld = xk.copy()
    if objective is not None : 
        loss = np.zeros(maxiters)

    for k in range(maxiters) : 
        xk = proximal(zkOld - 2/L*gradient(zkOld),norm,mu,L,n)
        tk = 0.5*(1 + np.sqrt(1 + 4*tkOld**2))
        zk = xkOld + (tkOld-1)/tk*(xk - xkOld)
        xkOld=xk.copy()
        zkOld=zk.copy()
        tkOld=tk
    if objective is not None : 
        loss[k] = objective(xk)
    if LA.norm(xk-xkOld) < eps and LA.norm(zk-zkOld) < eps:
        return xk
    return xk
