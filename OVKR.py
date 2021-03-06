## DATA IMPORTATION
#import os 
import scipy.io
import time
from opera.models import OVKR,OVKR_gridsearch
from opera.utils.plot import plot_err
from matplotlib import pyplot as plt
import numpy as np

#os.chdir("/home/lyx/")
mat = scipy.io.loadmat('simdata.mat')
X = mat.get('X')
y = mat.get('Y')

## OVKR TEST
obj = OVKR(muH=0.001,muC_1=0.001,normC='L1')


print "Standard tests were performed with this object : \n"
obj.getparameter(show=True)

# simple fit, predict and score example
print "\n\nWe adapt the object with data from 'simdata'\n"
print "\tX : %sx%s\n" % X.shape
print "\tY : %sx%s\n" % y.shape
print "fit's computation time ..."
t = time.time()
obj.fit(X,y)
elapse_time_fit = time.time()-t #2 s
yt = obj.predict(X)
score_tra = obj.score(X,y)#3e-06
print "... %dsec" % elapse_time_fit

print "cross validation score's computation time ..."
# crossvalidation score example
t = time.time()
score_cv = obj.crossvalidationscore(X, y, 5)
elapse_time_cv = time.time()-t #16 s
score_cv#0.56
print "... %dsec" % elapse_time_cv

print "the object is fitted\n\t training error     \t\t%2.5f\n\t testing error (cv)\t\t%2.5f\n\t sparsity of C      \t%2.2f" % (score_tra,score_cv,float((obj.C == 0).sum())/obj.C.size*100) + '%'


# grid example
params = {'gamma' : [0.1,1,10] , 
          'muH' : [0.1,0.5,0,3,4,5,6,7,8] ,
          'muC1' : [0.1,0.5,0,3,4,5,6,7,8] , 
          'muC2' : [0.1,0.5,0,3,4,5,6,7,8] , 
          'normC' : ["L1","mixed","elasticnet","sparsemixed"]  }
print "Grid tests were performed with this parameters : \n\t%s" % params 

print "grid search's computation time ..."
t = time.time()
mdl = OVKR_gridsearch(X, y, 5, params)
elapse_time_grid = time.time()-t #93 sec
print "... %dsec" % elapse_time_grid
print "The object selected is : "
mdl.getparameter(show=True)
print "fit score and crossvalidation score's computation time ..."
mdl.fit(X,y)
gst = mdl.score(X,y)# 0.00015
gscv = mdl.crossvalidationscore(X, y, 5)# 0.51
print "the object is fitted\n\t training error     \t%s\n\t testing error (cv)\t%s\n\t sparsity of C      \t%2.2f" % (gst,gscv,float((mdl.C == 0).sum()/mdl.C.size*100)) + '%'


# plot test
obj.setparam("normC","L1")
muCs = pow(2.,np.array([0.1,0.5,0,3,4,5,6,7,8]))
plot_err(obj,"muC1",muCs,X,y,True,xscale='log')
plt.show()


