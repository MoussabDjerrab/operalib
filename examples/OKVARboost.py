import scipy.io
import time
from opera.models import OKVARboost
from opera.utils.conditionalIndependence import conditionalIndependence
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as LA
from opera.utils import spectralradius
from opera.utils.AUC import AUC,calc_auc_pr
from opera.utils import vec
from opera import kernels
import scipy.linalg as sLA

#x = np.ones(10)
#x[0]=0
#(pred,labels)=(x,x)
#B = np.array([[1,2,5,4],[1,4,5,2],[4,6,3,5],[1,2,3,2]])
#y = np.array([1,2,3,4])
#os.chdir("/home/lyx/")

mat = scipy.io.loadmat('DREAM3_size10Ecoli1.mat')
data = mat.get('data')[0]
M_ref = np.abs(mat["Ref"])

#mdl = OKVARboost(muH = 0.001,muC = 1,gammadc=0,max_iter=100,gammatr=0.2)
mdl = OKVARboost(muH=1,muC=1,gammadc=0,max_iter=4,gammatr=0.2,eps=1e-2)
print mdl
#p = mdl.boosting(data[0])
#J = mdl.jacobian(data[0],p)
mdl.fit(data,print_step=True)
#A = mdl.boosting(data[0])
params = {'gammatr' : [0,1,10] , 
          'gammadc' : [0,1,10] ,
          'muH' : [0.1,0,1,10] ,
          'muC' : [0.1,0,1,10] , 
          'jacobian_threshold' : [0.6] ,
          'adj_matrix_threshold': [0.7] , 
          'max_iter' : [4]
          }

#mdl = grid_search(data, M_ref, parameters = params)



mdl.predict(data,0.1,0.5)
M = mdl.adj_matrix
mdl.score(data, M_ref,0.1,0.5)
#mdl.score(data,M_ref)

M_vec = vec(mdl.adj_matrix)
Mvec = vec(M_ref)

