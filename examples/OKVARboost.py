import scipy.io
import time
from opera.models.OKVARboost import OKVARboost
from opera.utils.conditionalIndependence import conditionalIndependence
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as LA
from opera.utils import spectralradius
from opera.utils.AUC import AUC,calc_auc_pr
from opera.utils import vec



#os.chdir("/home/lyx/")

mat = scipy.io.loadmat('DREAM3_size10Ecoli1.mat')
data = mat.get('data')[0]
M_ref = np.abs(mat["Ref"])


mdl = OKVARboost(muH=0.1,gammadc=0,max_iter=100,gammatr=0.2)
#p = mdl.boosting(data[0])
#J = mdl.jacobian(data[0],p)

mdl.fit(data)

mdl.score(data,M_ref,0.5,0.5)


mdl.predict(data,1,0.5)
M = mdl.adj_matrix
mdl.score(data,M_ref)

mdl.predict(data, 1,0.5)
M_vec = vec(mdl.adj_matrix)
Mvec = vec(M_ref)

