import scipy.io
import numpy as np
from opera.kernels import Kernel as kernel
from opera import constraint, loss
from opera.models import OKVAR

mat = scipy.io.loadmat('DREAM3_size10Ecoli1.mat')
data = mat.get('data')[0][0]
x = data[:,0]
M_ref = np.abs(mat["Ref"])

obj = OKVAR(kernel(),constraint("lasso", 1))
obj.fit(data)
yt = obj.predict(data)

obj.score(data[:,1:],data[:,:20])


obj.fit(data[:20,:],data[1:,:])


predict(x0,range(0,0.01,10))

for i in range(1000) :


for i in list