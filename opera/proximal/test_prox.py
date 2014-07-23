import scipy.io
import time
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as LA
from opera.kernels import trgauss
from opera.utils import spectralradius, vec
from pyprox import forward_backward, soft_thresholding
from opera.proximal import proximalGeneral,proximal

#os.chdir("/home/lyx/")
mat = scipy.io.loadmat('simdata.mat')
X = mat.get('X')[:20,:20]
y = vec(mat.get('Y'))[:400]
muH = 1
muX = 1
K = trgauss(X,X,1)
A = K.copy()
norm = "elasticnet"

L = 2* spectralradius(np.dot(K.T,K))
x0 =  np.linalg.solve(K + muH*np.identity(K.shape[0]), y)


def f(x,l=muX) :
    return proximal(x,norm=norm,mu1=l)


#l1-reg
def g(x):
    #if x.ndim < 1 : 
    #    return 0
    return np.dot(A.T, np.dot(A, x) - y)


x_pyprox = forward_backward(f, g, x0, L, maxiter=100, method='fista')
#LA.norm(x_pyprox)
x_opera = proximalGeneral(L,x0,g,maxiters=100,norm=norm,mu1=muX)
#LA.norm(x_opera)
(x_pyprox==x_opera).all()