import scipy.io
import time
from opera.models import OKVARboost
from opera.utils.conditionalIndependence import conditionalIndependence
from matplotlib import pyplot as plt
import numpy as np
from opera.kernels import Kernel
from opera.utils.plot import plot_predicted_graph

mat = scipy.io.loadmat('DREAM3_size10Ecoli1.mat')
data = mat.get('data')[0]
M_ref = np.abs(mat["Ref"])
l=[];
for i in range(10) : l.append("G"+str(i+1));
labels=np.array(l)

mdl = OKVARboost(kernel=Kernel(gammadc=0,gammatr=0.2,ovker="gram"),max_iter=4,eps=1e-2)

mdl.fit(data,print_step=False)

mdl.predict(data,0.45,0.65)
mdl.score(data, M_ref,0.45,0.65)
mdl.score(data,M_ref,0.6,0.8)

plot_predicted_graph(mdl.adj_matrix,M_ref,labels)

nA = 10
A = np.array(range(nA))*1./nA
(m,tj,ta)=(0,0,0)
for j in A :
    for a in A :
        (_,err) = mdl.score(data,M_ref,j,a)
        print (err,j,a)
        if err>m :
            (m,tj,ta)=(err,j,a)



