from .OVKR import OVKR
import numpy as np
from opera.utils.conditionalIndependence import conditionalIndependence
from opera import loss as Loss
import scipy.linalg as LA

class OKVAR(OVKR):
    """
    Performs OVK regression over parameter ranges, cross-validation, etc.

    :param kernel:
    :type opera.kernels.Kernel
    :param constraint:
    :type opera.constraint
    :param loss:
    :type opera.loss
    """

    def __init__(self, kernel, constraint, loss = None):

        if loss is None :
            loss = Loss()
            def f(x) :
                K = kernel.matrix()
                S = 0
                for t in range(x.shape[0]-1) :
                    S += LA.norm(x[t+1,:]-np.dot(K,x[t,:]),2)**2
                return S
            def fprime(x):
                K = kernel.matrix()
                S = 0
                for t in range(x.shape[0]-1) :
                    S += np.dot(K,np.dot(K,x[t,:])-x[t+1,:])
                return S
            loss.add_function(f,fprime)
        super(OVKR,self).__init__(kernel, constraint, loss)


    def __repr__(self):
        if self.kernel.K is None : fitted = "no "
        else : fitted = "yes"
        return "OKVAR model : < fitted:"+fitted+" >"
    def __str__(self):
        out = "OKVAR model : \n"
        #parameters print
        out += str(self.kernel)+"\n"
        out += str(self.constraint)+"\n"
        out += str(self.loss)+"\n"
        return out
    def copy(self):
        return OKVAR(self.kernel.copy(),self.constraint.copy(),self.loss.copy())

    def fit(self,TS,allvec=False):
        if allvec :
            X = TS.copy()
            y = X.copy()
        else :
            n = TS.shape[0]
            X = TS[:n-1,:]
            y = TS[1:,:]
        OVKR.fit(self, X, y)

