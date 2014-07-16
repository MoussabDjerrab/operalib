import scipy.io
import time
from opera.models.OKVARboost import OKVARboost
from opera.utils.conditionalIndependence import conditionalIndependence
from matplotlib import pyplot as plt
import numpy as np

#os.chdir("/home/lyx/")
mat = scipy.io.loadmat('simdata.mat')
X = mat.get('X')
y = mat.get('Y')


mat = scipy.io.loadmat('DREAM3_size10Ecoli1.mat')
data = mat.get('data')[0]

mat = scipy.io.loadmat('samples.mat')

mdl = OKVARboost(muH=0.1,gammadc=0,max_iter=50,gammatr=1)
mdl.fit(data[0])
J = mdl.predict(data[0][1:])
