import numpy as np
from proximal import proximal
import numpy.linalg as LA

def proximalGeneral(L,init=None,gradient=None,objective=None,print_objective=False,maxiters=100,norm='L1',mu1=1,mu2=1,partition=None,weight_partition=None,eps=1.e-3,backtrack=False,debug=False):
    """
    init : initial vector (must be given)
    L : Lispsitch coefficient
    objective : our function score
    maxiters : number of iteration of our algorithm
    norm : regularization norm
    mu1, mu2, partition, partition_weight : regularization parameters, see proximal for more informations
    eps : stop criterian
    """
    
    if init is None : 
        print "FISTA algorithm need an initial vector"
        return None
    if gradient is None : 
        print "FISTA algorithm need a gradient function"
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
        if backtrack :
            #TODO 
            L=L
        #
        #
        xkOld = xk.copy() 
        if debug : 
            print("step_"+str(k)+"\n\t step : "+str(mu1/L)+"\n\t norm of Yk : "+str(LA.norm(yk))+"\n\t grad of Yk : "+str(LA.norm(gradient(yk)))+"\n\t objective : "+str(objective(yk)))
        #   
        xk = proximal(yk-mu1/L*gradient(yk),  norm,mu1=mu1/L,mu2=1,partition=partition,weight_partition=weight_partition)
        #
        tkOld = tk
        tk = 0.5*(1 + np.sqrt(1 + 4*tkOld**2))
        #
        ykOld = yk.copy()
        yk = xk + ((tkOld-1)/tk)*(xk - xkOld)
        #
        if objective is not None : 
            loss[k] = objective(xk)
        #   
        if LA.norm(xk-xkOld) < eps and LA.norm(yk-ykOld) < eps:
            return xk
    
    return xk