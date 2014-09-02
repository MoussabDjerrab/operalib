from dcgauss import dcgauss
from dcpoly  import dcpoly
from dclin   import dclin
from trgauss import trgauss
from trpoly  import trpoly
from trlin   import trlin
from gramMatrix import f1_gramMatrix as gramMatrix
import numpy as np

class Kernel():
    def __init__(self,ovker="dc",ker="gauss",c=1,d=3,gamma=1,B="identity",gammadc=None,gammatr=None) :
        """ Choose a self.ker function
        n is the number of column of y
        self.ovker is :
            dc : transformable : K(x,z)_(ij) = k(xi,zi)
            tf : decomposable : K(x,z) = B x k(x,z)
            custom : the user give a K
        self.ker is :
            gauss : gaussian self.ker
            linear : linear self.ker
            poly : polunomial self.ker
            custom : the user give a k
        other parameter :
            c and d for linear self.ker (default c=1 and d=3)
            g for gaussian self.ker (default g=1)
            B for decomposable type (default B='identity')
                identity : B is id
                cov : B is cov(Y)
                learn : the algo learn B
                custom : the user give a B
        """
        self.ovker = ovker
        self.ker = ker
        self.c = c
        self.d = d
        self.gamma = gamma
        self.B = B
        if gammadc is None : self.gammadc = gamma
        else : self.gammadc = gammadc
        if gammatr is None : self.gammatr = gamma
        else : self.gammatr = gammatr
        self.K = None
        self.f = fun_matrix(self)
    def copy(self):
        ker = Kernel(ovker=self.ovker,ker=self.ker,c=self.c,d=self.d,gamma=self.gamma,B=self.B)
        ker.K = self.K
        ker.f = self.f
        return ker

    def compute_matrix(self,X1,X2,y,keep_in_memory=True):
        K = self.f(X1,X2,y)
        if keep_in_memory : self.K = K
        return K

    def matrix(self):
        if self.K is None :
            print("Error in Kernel, there is no matrix")
        return self.K

def fun_matrix(obj):
    if obj.ovker=="dc" or obj.ovker == None :
        if obj.ker == "gauss" or obj.ker == None :
            def f(X1,X2,y): return dcgauss(X1,X2,obj.gamma,createB(y,obj.B))
        elif obj.ker == "linear" :
            def f(X1,X2,y): return dclin(X1,X2,createB(y,obj.B))
        elif obj.ker == "polynomial" :
            def f(X1,X2,y): return dcpoly(X1,X2,obj.c,obj.d,createB(y,obj.B))
        elif obj.ker.__class__ == np.ndarray :
            def f(X1,X2,y): return obj.ker.copy
    elif obj.ovker == "tr" :
        if obj.ker == "gauss"  or obj.ker == None :
            def f(X1,X2,y): return trgauss(X1,X2,obj.gamma)
        elif obj.ker == "linear" :
            def f(X1,X2,y): return trlin(X1,X2)
        elif obj.ker == "polynomial" :
            def f(X1,X2,y): return trpoly(X1,X2,obj.c,obj.d)
        elif obj.ker.__class__ == np.ndarray :
            def f(X1,X2,y): return obj.ker.copy()
    elif obj.ovker == "mixed" or obj.ovker == "gram":
        if obj.ker == "gauss" or obj.ker == None :
            def f(X1,X2,y): return gramMatrix(X1,X2,createB(y,obj.B),obj.gammadc,obj.gammatr)
    elif obj.ovker.__class__ == np.ndarray :
        def f(X1,X2,y): return obj.ovker.copy()
    return f

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
    elif B.shape[0]==B.shape[1] and B.shape[0]==y.shape[0] :
        B = B.copy()
    elif B.__class__ is np.ndarray :
        B = B.copy()
    else : print "Error in self.ker, B is not in the good format"
    #elif B == "learn" :
        #B = self.kers.learn()
    return B
