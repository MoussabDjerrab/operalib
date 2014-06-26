## DATA IMPORTATION
import os 
import scipy.io
import time
from opera.models import OVKR,OVKR_gridsearch

os.chdir("/home/lyx/")
mat = scipy.io.loadmat('simdata.mat')
X = mat.get('X')
y = mat.get('Y')

## OVKR TEST



obj = OVKR(normC="mixed",muH=0.001)
# simple fit, predict and score example
t = time.time()
obj.fit(X,y)
elapse_time_fit = time.time()-t #11.25 s
yt = obj.predict(X)
obj.score(X,y)#0.001


# crossvalidation score example
t = time.time()
score_cv = obj.crossvalidationscore(X, y, 5)
elapse_time_cv = time.time()-t #16 s
score_cv#0.56



# grid example
params = {'gamma' : [0.1,1,10] , 
          'muH' : [0.001] ,
          'muC' : [0.001] , 
          'normC' : ["L1","mixed"]  }
t = time.time()
mdl = OVKR_gridsearch(X, y, 5, params)
elapse_time_grid = time.time()-t #96 sec
mdl.fit(X,y)
mdl.score(X,y)# 0.0002
mdl.crossvalidationscore(X, y, 5)# 0.54
mdl.gamma   ,mdl.muH    ,mdl.muC    ,mdl.normC
#(0.1, 0.001, 0.001, 'L1')

