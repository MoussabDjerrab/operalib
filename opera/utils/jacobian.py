import numpy as np
import opera.kernels.trgauss as trgauss
import scipy.linalg as sLA


def jacobian(obj,data,param):
    """Method to compute a jacobian on a model after a boosting


    Parameters
    ----------
        data : ndarray [n,d] where n is the number of samples and d is the number of features.

        params : dictionnary
            see :func:'utils.boosting()'
    """
    nb_iter = param.size

    if param.ndim>0 : nFrac = param[0].subset[0].size
    (N,p) = data.shape
    tf = N - 1 #number of time points when predictions are made
    mStop = nb_iter
    Js = np.array([None] * mStop)
    J = np.zeros((p,p))
    data = data[0:tf,:]
    #TODO : outside this parameter
    betaParam = obj.beta; # diffusion parameter for the Laplacian

    for m in range(mStop) :
        W_m = param[m].W
        C_m = param[m].C #; np.reshape(self.param[m].C,(p,tf))# just to make sure it is correct size
        #
        W_m_sub = W_m[np.ix_(param[m].subset,param[m].subset)]
        C_m_sub = C_m[param[m].subset,:]
        #
        deg_W = np.diag(np.sum(W_m_sub,0))
        L_m_sub = deg_W - W_m_sub #standard Laplacian
        #
        B_m_sub = sLA.expm(betaParam*L_m_sub);
        B_m = np.zeros((p,p));
        B_m[np.ix_(param[m].subset,param[m].subset)] = B_m_sub;
        #
        tmpJ_sub = np.zeros((nFrac,nFrac))
        tmpJ = np.zeros((p,p))
        # iterate over time points 1:tf
        for t in range(tf):
            for l in range(tf) :
                #TODO : K_lt when gammadc not null
                K_lt = (trgauss.trgauss(data[l,param[m].subset].T,data[t,param[m].subset].T,obj.kernel.gammatr))
                tmpJ1 = np.tile(data[l,param[m].subset],(nFrac,1))-np.tile(data[t,param[m].subset],(nFrac,1))
                tmpJ2 = np.tile(C_m_sub[:,l],(nFrac,1))
                tmpJ_sub = tmpJ_sub + tmpJ1*K_lt*tmpJ2

        tmpJ[np.ix_(param[m].subset,param[m].subset)] = tmpJ_sub

        #
        J = J + (2./tf)*obj.kernel.gammatr*param[m].rho*B_m*tmpJ
        Js[m] = J
    return J

def jacobian_uc(obj,data,param):
    """Method to compute a jacobian on a model after a boosting

    Parameters
    ----------
        data : ndarray [n,d] where n is the number of samples and d is the number of features.

        params : dictionnary
            see :func:'utils.boosting()'
    """
    nb_iter = param.shape[0]

    nFrac = param[0].subset.size
    (N,p) = data.shape
    tf = N - 1 #number of time points when predictions are made
    mStop = nb_iter
    Js = np.array([None] * mStop)
    J = np.zeros((p,p))
    data = data[0:tf,:]
    #TODO : outside this parameter

    for m in range(mStop) :
        W_m = param[m].W
        C_m = param[m].C #; np.reshape(self.param[m].C,(p,tf))# just to make sure it is correct size
        #
        W_m_sub = W_m[np.ix_(param[m].subset,param[m].subset)]
        C_m_sub = C_m[param[m].subset,:]
        #
        deg_W = np.diag(np.sum(W_m_sub,0))
        L_m_sub = deg_W - W_m_sub #standard Laplacian
        #
        B_m_sub = sLA.expm(param[m].beta*L_m_sub);
        B_m = np.zeros((p,p));
        B_m[np.ix_(param[m].subset,param[m].subset)] = B_m_sub;
        #
        tmpJ_sub = np.zeros((nFrac,nFrac))
        tmpJ = np.zeros((p,p))
        # iterate over time points 1:tf
        for t in range(tf):
            for l in range(tf) :
                #TODO : K_lt when gammadc not null
                K_lt = (trgauss(data[l,param[m].subset].T,data[t,param[m].subset].T,obj.kernel.gammatr))
                tmpJ1 = np.tile(data[l,param[m].subset],(nFrac,1))-np.tile(data[t,param[m].subset],(nFrac,1))
                tmpJ2 = np.tile(C_m_sub[:,l],(nFrac,1))
                tmpJ_sub = tmpJ_sub + tmpJ1*K_lt*tmpJ2

        tmpJ[np.ix_(param[m].subset,param[m].subset)] = tmpJ_sub
        #
        J = J + (2./tf)*obj.kernel.gammatr*param[m].rho*B_m*tmpJ
        Js[m] = J
    return J
