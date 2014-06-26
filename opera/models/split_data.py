import numpy as np

def __init__(X,y,coef_test=1,coef_train=4):
    """ 
    split data in two set of data. one of size coef_test/(coef_test+coef_train), the other in coef_test/(coef_test+coef_train)
    """
    n = coef_test+coef_train
    #our separation
    s = max([X.shape[0]/n+1,1])
    indices = np.random.permutation(X.shape[0])
    blocs=[]
    
    i = 0
    while (i+1)*s <= indices.shape[0] : 
        idx = indices[i*s:(i+1)*s]
        if X.ndim == 1 : 
            Xbloc = X[idx]
        else :
            Xbloc = X[idx,:]
        if y.ndim == 1 : 
            ybloc = y[idx]
        else :
            ybloc = y[idx,:]
        
        blocs.append((Xbloc,ybloc))
        i = i+1
    return blocs