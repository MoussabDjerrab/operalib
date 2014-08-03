import numpy as np
import opera.kernels as kernels
import scipy.linalg as sLA


def jacobian(obj,data,param):
    """Method to compute a jacobian on a model after a boosting
    
    
    Parameters
    ----------      
        data : ndarray [n,d] where n is the number of samples and d is the number of features.
        
        params : dictionnary
            see :func:'utils.boosting()'
    """
    W = param["W"] 
    C = param["C"] 
    subset = param["subset"] 
    nb_iter = param["nb_iter"]
    rho = param["rho"]
    
    nFrac = subset[0].size
    (N,p) = data.shape
    tf = N - 1 #number of time points when predictions are made 
    mStop = nb_iter
    Js = np.array([None] * mStop)
    J = np.zeros((p,p))
    data = data[0:tf,:]
    #TODO : outside this parameter
    betaParam = .2; # diffusion parameter for the Laplacian

    for m in range(mStop) :
        W_m = W[m]
        C_m = C[m] #; np.reshape(self.C[m],(p,tf))# just to make sure it is correct size
        #
        W_m_sub = W_m[np.ix_(subset[m],subset[m])]
        C_m_sub = C_m[subset[m],:]
        #
        deg_W = np.diag(np.sum(W_m_sub,0))
        L_m_sub = deg_W - W_m_sub #standard Laplacian
        #
        B_m_sub = sLA.expm(betaParam*L_m_sub);
        B_m = np.zeros((p,p));
        B_m[np.ix_(subset[m],subset[m])] = B_m_sub;
        #
        tmpJ_sub = np.zeros((nFrac,nFrac))
        tmpJ = np.zeros((p,p))
        # iterate over time points 1:tf
        for t in range(tf):
            for l in range(tf) :
				#TODO : K_lt when gammadc not null
                K_lt = (kernels.trgauss(data[l,subset[m]].T,data[t,subset[m]].T,obj.gammatr))
                tmpJ1 = np.tile(data[l,subset[m]],(nFrac,1))-np.tile(data[t,subset[m]],(nFrac,1))
                tmpJ2 = np.tile(C_m_sub[:,l],(nFrac,1))
                tmpJ_sub = tmpJ_sub + tmpJ1*K_lt*tmpJ2

        tmpJ[np.ix_(subset[m],subset[m])] = tmpJ_sub

        #
        J = J + (2./tf)*obj.gammatr*rho[m]*B_m*tmpJ
        Js[m] = J 
    return J
