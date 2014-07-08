import scipy.io
import time
from opera.models import OVKR,OVKR_gridsearch
from opera.utils.conditionalIndependence import conditionalIndependence
from matplotlib import pyplot as plt
import numpy as np

#os.chdir("/home/lyx/")
mat = scipy.io.loadmat('simdata.mat')
X = mat.get('X')
y = mat.get('Y')

(r,s) = conditionalIndependence(X, 0)
