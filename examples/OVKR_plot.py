## DATA IMPORTATION
from opera.models import OVKR
from matplotlib import pyplot as plt
import numpy as np
from numpy.matlib import rand
from opera import constraint, loss
from opera.kernels import Kernel as kernel

"""
the space is in [-5,5]^2
"""
N = 1000

def isSup(x,y):
    return x**2+y**2<4**2 and x**2+y**2>2**2 and y>0.5

def isInf(x,y):
    return x**2+y**2<4**2 and x**2+y**2>2**2 and y<-0.5

def isCir(x,y):
    return x**2+y**2<1.6**2

data = np.array((rand((N,2))*10-5))
labels = np.ones((N,2))

for i in range(N) :
    x = data[i][0]
    y = data[i][1]
    if( isCir(x,y)) : labels[i] = np.array([0,0])
    if( isInf(x,y)) : labels[i] = np.array([0,1])
    if( isSup(x,y)) : labels[i] = np.array([1,0])


def color(label):
    if (label==[0,0]).all() : return "ko"
    if (label==[0,1]).all() : return "wo"
    if (label==[1,0]).all() : return "ro"
    if (label==[1,1]).all() : return "yo"


def plot(data,labels):
    for i in range(N) :
        x = data[i][0]
        y = data[i][1]
        plt.plot(x,y,color(labels[i]))

#plot(data,labels)
#plt.show()



obj = OVKR(kernel(),constraint("lasso", 1),loss())
obj.fit(data,labels)
n_labels = (obj.predict(data)>=0.5)
#plot(data,n_labels)
#plt.show()
s = obj.score(data,labels)
s


