import numpy as np
from proximal import proximal
import numpy.linalg as LA

def proximalGeneral(L,init=None,objective=None,maxiters=100,norm='L1',mu=1,n=1,eps=1.e-3):
    if init is None : 
        print "FISTA algorithm need an initial vector"
        return None
    
    # step0 : y1=x0, t1=1
    tk = 1
    yk = init.copy()
    xk = init.copy()
    if objective is not None : 
        loss = np.zeros(maxiters)
        
        
    for k in range(maxiters) : 
        #step k
        #    xk = pl(yk)
        #    t(k+1) = ( 1+sqrt(1+4tk^2) ) / 2
        #    y(k+1) = xk + (tk-1/t(k+1))*(xk-x(k-1))
        
        xkOld = xk.copy() 
        xk = proximal(yk,norm,mu,L,n)
        
        tkOld = tk
        tk = 0.5*(1 + np.sqrt(1 + 4*tkOld**2))
        
        ykOld = yk.copy()
        yk = xk + ((tkOld-1)/tk)*(xk - xkOld)
        
        if objective is not None : 
            loss[k] = objective(xk)
            
        if LA.norm(xk-xkOld) < eps and LA.norm(yk-ykOld) < eps:
            return xk
    
    return xk