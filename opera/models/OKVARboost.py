"""
.. module:: OKVARboost
   :platform: Unix, Windows
   :synopsis: module to performs an OKVARboost

.. moduleauthor:: Tristan Tchilinguirian <tristan.tchilinguirian@ensiie.fr>

Reverse engineering of gene regulatory networks remains a central challenge in computational systems biology, despite recent advances facilitated by benchmark in silico challenges that have aided in calibrating their performance.
Nonlinear dynamical models are particularly appropriate for this inference task, given the generation mechanism of the time-series data. We have introduced a novel nonlinear autoregressive model based on operator-valued kernels.
A flexible boosting algorithm (OKVAR-Boost) that shares features from L2-boosting and randomization-based algorithms is developed to perform the tasks of parameter learning and network inference for the proposed model.

    * Lim et al., (2013) OKVAR-Boost: a novel boosting algorithm to infer nonlinear dynamics and interactions in gene regulatory networks. Bioinformatics 29 (11):1416-1423.
"""

from .OPERAObject import OPERAObject
import numpy as np
from opera.utils import AUC
from scipy.stats.mstats import mquantiles as quantile
from opera.utils import vec, jacobian
from opera.boosting import boosting

def grid_search(data,M,score="AUPR",parameters={},print_step=False):
    """
    Do a search of the best choices of parameter by minimizing the AUPR score with nblocks blocks
    """
    if score.upper() == "AUROC" :
        s = 0
    else :
        s = 1

    bestscore = 0
    bestmodel = None
    gammadcs=[1e-4]
    gammatrs=[1]
    muHs=[0.001]
    muCs=[1]
    randFracs=[1]
    alphas=[0.05]
    n_edge_picks=[0]
    epss=[1e-4]
    max_iters=[100]
    jacobian_thresholds = [0.95]
    adj_matrix_thresholds = [0.5]

    if parameters.has_key('gammadc') :
        gammadcs = parameters['gammadc']
    if parameters.has_key('gammatr') :
        gammatrs = parameters['gammatr']
    if parameters.has_key('muH') :
        muHs = parameters['muH']
    if parameters.has_key('muC') :
        muCs = parameters['muC']
    if parameters.has_key('randFrac') :
        randFracs = parameters['randFrac']
    if parameters.has_key('alpha') :
        alphas = parameters['alpha']
    if parameters.has_key('n_edge_pick') :
        n_edge_picks = parameters['n_edge_pick']
    if parameters.has_key('eps') :
        epss = parameters['eps']
    if parameters.has_key('max_iter') :
        max_iters = parameters['max_iter']
    if parameters.has_key('jacobian_threshold') :
        jacobian_thresholds = parameters['jacobian_threshold']
    if parameters.has_key('adj_matrix_threshold') :
        adj_matrix_thresholds = parameters['adj_matrix_threshold']


    for gammadc in gammadcs :
        for gammatr in gammatrs :
            for muH in muHs :
                for muC in muCs :
                    for randFrac in randFracs :
                        for alpha in alphas :
                            for n_edge_pick in n_edge_picks :
                                for eps in epss :
                                    for max_iter in max_iters :
                                        for jacobian_threshold in jacobian_thresholds :
                                            for adj_matrix_threshold in adj_matrix_thresholds:
                                                obj = OKVARboost(gammadc,gammatr,muH,muC,randFrac,alpha,n_edge_pick,eps,max_iter)
                                                obj.fit(data)
                                                score = obj.score(data,M,jacobian_threshold,adj_matrix_threshold)[s]
                                                if print_step : print obj
                                                if( score > bestscore) :
                                                    bestmodel = obj
                                                    bestscore = score
    return bestmodel




class OKVARboost(OPERAObject):
    """
    .. class:: OKVARboost

    This implements OKVARboost fitting, prediction and score


    :param muH: ridge penalty parameter (lambda2)
    :type muH: float , default=0.001
    :param muC: l1 penalty parameter (lambda1)
    :type muC: float , default=1.
    :param gammadc: Parameter of decomposable gaussian matrix-valued kernel
    :type gammadc: float , default=0.
    :param gammatr: Parameter of transformable gaussian matrix-valued kernel
    :type gammatr: float , default=1.
    :param alpha: Level of the partial correlation test is set to a conservative
    :type alpha: float , default=1.
    :param eps: Stopping criterion threshold for the norm of the residual vector
    :type eps: float , default=1.e-4
    :param max_iter: Number of boosting iterations
    :type max_iter: int , default=100
    :param randFrac: Size of random subset as a percentage of the network size
    :type randFrac: float in [0,1] , default=1.
    :param n_edge_pick: Number of edges to pick in each random subset. If 0 then all the significant edges are picked
    :type n_edge_pick: int , default=0
    :param flagRes: If it is True then variables whose residuals are too low are removed at each iteration
    :type flaRes: bool , default=True
    """
    gammadc = 1e-4
    gammatr = 1
    muH = 0.001
    muC = 1
    randFrac = 1
    alpha = 0.05
    n_edge_pick = 0
    eps = 1e-4
    max_iter = 100
    flagRes = 1


    def __init__(self,gammadc=1e-4,gammatr=1,muH=0.001,muC=1,randFrac=1,alpha=0.05,n_edge_pick=0,eps=1e-4,flagRes=True,max_iter=100):
        self.gammadc = gammadc
        """Parameter of decomposable gaussian matrix-valued kernel"""
        self.gammatr = gammatr
        """Parameter of transformable gaussian matrix-valued kernel"""
        self.muH = muH
        """ridge penalty parameter (lambda2)"""
        self.muC = muC
        """l1 penalty parameter (lambda1)"""
        self.randFrac = randFrac
        """Size of random subset as a percentage of the network size"""
        self.alpha = alpha
        """Level of the partial correlation test is set to a conservative"""
        self.n_edge_pick = n_edge_pick
        """ Number of edges to pick in each random subset. If 0 then all the significant edges are picked"""
        self.eps = eps
        """Stopping criterion threshold for the norm of the residual vector"""
        self.flagRes = flagRes
        """If it is True then variables whose residuals are too low are removed at each iteration"""
        self.max_iter = max_iter
        """Number of boosting iterations"""
        self.boosting_param = None
        """returns of fit , see :func:`boosting.boosting()`"""
        self.adj_matrix = None
        """returns of predict"""
    def __repr__(self):
        if self.boosting_param is None : fitted = "no "
        else : fitted = "yes"
        if self.adj_matrix is None : predicted = "no "
        else : predicted = "yes"
        return "OKVARboost model : < fitted:"+fitted+" | predicted:"+predicted+" >"
    def __str__(self):
        out = "OKVARboost model : \n\t Hyperparameters :\n"
        #parameters print
        def item(s,r):
            return "\t\t"+s+str(r)+"\n"
        out+=item("ridge penalty parameter (lambda2) : muH=",self.muH)
        out+=item("l1 penalty parameter (lambda1) : muC=",self.muC)
        out+=item("Parameter of decomposable gaussian matrix-valued kernel : gamma_dc=",self.gammadc)
        out+=item("Parameter of transformable gaussian matrix-valued kernel : gamma_tr=",self.gammatr)
        out+=item("Level of the partial correlation test is set to a conservative : alpha=",self.alpha)
        out+=item("Stopping criterion threshold for the norm of the residual vector : eps=",self.eps)
        out+=item("Number of boosting iterations : ",self.max_iter)
        out+=item("Size of random subset as a percentage of the network size : randFrac=",self.randFrac)
        if self.n_edge_pick > 0 : item("Number of edges to pick in each random subset : n_edge_pick=",self.n_edge_pick)
        else :  item("Number of edges to pick in each random subset : n_edge_pick=","all the significant edges")
        if self.flagRes : f = "yes"
        else : f = "no"
        item("Variables whose residuals are too low are removed at each iteration : flagRes=",f)
        #fit print
        if self.boosting_param is None : s = "the model is not fitted yet"
        else : s="the model is fitted with "+str(self.boosting_param.size)+" time series"
        out+="\t"+s+"\n"
        #predicted print
        if self.adj_matrix is None : out+="\tthe model is not predicted yet\n"
        else :
            out+="\tthe model is predicted with : \n"
            out+=item("jacobian threshold : ",self.jacobian_threshold)
            out+=item("consensus threshold : ",self.adj_matrix_threshold)
            if self.auroc is not None :  out+="\tthe score of our model is : \n"+item("AUROC : ",self.auroc)+item("AUPR : ",self.aupr)
        return out


    def fit(self,data,print_step=False):
        """Method to fit a model

        :param data: Cell of N array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        :type data: ndarray [N,n,d]
        :param print_step: If it is true then displayed to the user the current step on the standard output
        :type: bool , default=False

        :returns:  boosting_param (ndarray of dictionnary) as an attribute. For each time serie it compute a dictionnary, see :func:`boosting.boosting` for more information about boosting_param
        """
        N = data.size
        params = np.array([None]*N)
        for i in range(N) :
            if print_step :
                print "data no "+str(i)
            params[i] = boosting(self,data[i],print_step=print_step)
        self.boosting_param = params

    def predict(self,data,jacobian_threshold=0.50,adj_matrix_threshold=0.50):
        """Method to predict a model

        :param data: Cell of N array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        :type data: ndarray [N,n,d]
        :param jacobian_threshold: Quantile level of the Jacobian values used to get the adjacency matrix
        :type jacobian_threshold: float , default=0.5
        :param adj_matrix_threshold: Quantile level of the adjacency matrix valued used to get the final adjacency matrix
        :type adj_matrix_threshold: float , default=0.5

        :returns: Adjacency matrix of our datas
        :rtype: nparray
        """
        #we save the threshold for the print
        self.jacobian_threshold=jacobian_threshold
        self.adj_matrix_threshold=adj_matrix_threshold
        self.auroc = None
        (_,p) = data[0].shape
        M = np.zeros((p,p))
        params = self.boosting_param
        for i in range(data.size) :
            Ji = np.abs(np.tanh(jacobian(self,data[i], params[i])))
            delta = quantile(vec(Ji),jacobian_threshold)
            M = M+(Ji>=delta)
        delta = quantile(vec(M),adj_matrix_threshold)
        M = (M>=delta)
        self.adj_matrix=M*1

		#TODO
		#Pour chaque data{i} :
		#	fit and predict <- 10fois
		#	add and threshold

    def score(self, data, M, jacobian_threshold=1,adj_matrix_threshold=0.50):
        """Method to give the AUROC and AUPR score a model

        :param data: Cell of N array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        :type data: ndarray [N,n,d]
        :param jacobian_threshold: Quantile level of the Jacobian values used to get the adjacency matrix
        :type jacobian_threshold: float , default=0.5
        :param adj_matrix_threshold: Quantile level of the adjacency matrix valued used to get the final adjacency matrix
        :type adj_matrix_threshold: float , default=0.5

        :returns: The AUROC and AUPR score of our model with M as true matrix
        :rtype: (float,float)
        """
        if (self.adj_matrix is None) or not(self.jacobian_threshold==jacobian_threshold) or not(self.adj_matrix_threshold==adj_matrix_threshold) :
            self.predict(data, jacobian_threshold, adj_matrix_threshold)
        M_vec = np.reshape(self.adj_matrix,self.adj_matrix.size)
        Mvec = np.reshape(M,M.size)
        (self.auroc,self.aupr) = AUC(M_vec,Mvec)
        return (self.auroc,self.aupr)


	#def plot_adjacency_matrix

