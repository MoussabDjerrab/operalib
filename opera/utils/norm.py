import numpy.linalg as LA
import numpy as np



def norm(x,norm="l1",mu=1):
    """
    norm is the Omega function in 
    http://arxiv.org/pdf/1108.0775.pdf - 3.A Principle of Proximal error
    mu = lamba/L     
    """
    if norm.upper() == "L1" : 
        return LA.norm(x,1)
    elif norm.upper() == "L2" : 
        return (1/(1+mu)) * LA.norm(x,2)
    elif norm.lower() ==  "elasticnet":
        return norm(x, norm="L1") + norm(x, norm="L2")
    elif norm.lower() ==  "grouplasso":
        s = 0
        for i in range(len(x[0])) : 
            s = s + x[i]
        return 
    elif norm.lower() ==  "sparsegrouplasso":
        return norm(x, norm="L1") + norm(x, norm="grouplasso")
    elif norm.upper() == "mixed" : 
        return np.mean(norm(x, norm="L1"),norm(x, norm="L2"))
