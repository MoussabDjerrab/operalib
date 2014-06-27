import matplotlib.pyplot as plt
import numpy as np

def plot_err(obj,var,ensvar,X,y,sparcity=False):
    ensvar.sort()
    def f(x) : 
        obj.setparam(var,x)
        obj.fit(X,y)
        tra = obj.score(X,y)
        tes = obj.crossvalidationscore(X,y)
        spar = float((obj.C == 0).sum())/obj.C.size
        return (tra,tes,spar)
    train_err = []
    valid_err = []
    sparc  = []
    for x in ensvar : 
        (tra,tes,spar) = f(x)
        train_err.append(tra)
        valid_err.append(tes)
        sparc.append(spar)
    plt.plot(ensvar,train_err,'x-')
    plt.plot(ensvar,valid_err,'o-')
    if sparcity : 
        plt.plot(ensvar,sparc,'s-')
        plt.legend(['training error','testing error','sparcity of C (proportion)'],'best')
    else : 
        plt.legend(['training error','testing error'],'best')
    
    xmin = ensvar[(np.array(valid_err)).argmin()]
    ymin = (np.array(valid_err)).min()
    (axm,_,aym,_)  = plt.axis()
    plt.plot([xmin,xmin],[aym,ymin],'--b')
    plt.plot([axm,xmin],[ymin,ymin],'--b')
    plt.xlabel(var)
    plt.ylabel("error")
    return xmin
    
