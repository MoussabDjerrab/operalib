import numpy as np
import numpy.linalg as LA
from opera.utils import spectralradius
from opera import loss
import opera.constraint
from shootingLinear import elastic_shooting

# TODO : Voir en details pyprox pour voir si on peut utiliser leurs librairies

def proximal(Loss,Constraint,init,L,maxiters=100,eps=None,backtrack=False,debug=False):

    def gradient(x):
        grad = 0
        for f in Loss.gradients() :
            grad += f(x)
        for g in Constraint.gradients() :
            grad += g(x)
        return grad
    if (Loss.gradients().size+Constraint.gradients().size)==0 :
        print "FISTA algorithm need a gradient function"
        return None

    def objective(x):
        obj = 0
        for f in Loss.functions() :
            obj += f(x)
        for (g,_) in Constraint.functions() :
            obj += g(x)
        return obj

    if eps is None :
        eps = LA.norm(init)*1e-5

    prox = Constraint.prox_operator()

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
            print("step_"+str(k)+"\n\t step : "+str(1/L)+"\n\t norm of Yk : "+str(LA.norm(yk))+"\n\t grad of Yk : "+str(LA.norm(gradient(yk))))
        #
        xk = prox(yk-1/L*gradient(yk),1/L)
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

def proximalLinear(K, y, Constraints, Loss=None,init=None, maxiters=100, eps=None):
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

    lambda2 = Constraints.lambda2

    if init is None :
        init = np.linalg.solve(K + lambda2*np.identity(K.shape[0]), y)

    if Loss is None :
        Loss = loss()
        f = lambda x : LA.norm(y-np.dot(K,x),2)**2
        fprime = lambda x : np.dot(K,np.dot(K,x)-y)
        Loss.add_function(f,fprime)
    Constraints.K = K

    return proximal(Loss, Constraints, init, L, maxiters, eps)

