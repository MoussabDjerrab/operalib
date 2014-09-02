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
from opera.kernels import Kernel
import scipy.linalg as sLA
import numpy.linalg as LA
from opera.utils.conditionalIndependence import conditionalIndependence
from opera.models import OKVAR
from opera.constraint import constraint as Constraint
from opera.proximal import proximalLinear
import opera.kernels.gramMatrix
from opera.boosting import boosting

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

    def __init__(self,kernel=None,constraint=None,loss=None,randFrac=1,alpha=0.05,tot=1000,beta=0.2,n_edge_pick=0,eps=1e-4,flagRes=True,max_iter=4):
        self.loss = loss
        if kernel is None : self.kernel = Kernel("gram")
        else : self.kernel = kernel
        if constraint is None : self.constraint = Constraint(reg="elasticnet")
        else : self.constraint = constraint
        self.boost = None
        self.adj_matrix = None
        self.maxiter = max_iter
        self.nFrac= randFrac
        self.alpha = alpha
        self.n_edge_pick = n_edge_pick
        self.flagRes = flagRes
        self.eps = eps
        self.tot = tot #1000 #maximum number of trials for the partial correlation test
        self.beta = beta #0.2 # diffusion parameter for the Laplacian

    def __repr__(self):
        if self.boost is None : fitted = "no "
        else : fitted = "yes"
        if self.adj_matrix is None : predicted = "no "
        else : predicted = "yes"
        return "OKVARboost model : < fitted:"+fitted+" | predicted:"+predicted+" >"
    def __str__(self):
        out = "OKVARboost model :\n"
        return out

    def fit(self,data,print_step=False):
        """Method to fit a model
        :param data: Cell of N array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        :type data: ndarray [N,n,d]
        :param print_step: If it is true then displayed to the user the current step on the standard output
        :type: bool , default=False
        :returns: boosting_param (ndarray of dictionnary) as an attribute. For each time serie it compute a dictionnary, see :func:`boosting.boosting` for more information about boosting_param
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

    def fit_uc(self,data,print_step=False):
        """Method to fit a model

        :param data: Cell of N array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        :type data: ndarray [N,n,d]
        :param print_step: If it is true then displayed to the user the current step on the standard output
        :type: bool , default=False

        :returns:  boosting_param (ndarray of dictionnary) as an attribute. For each time serie it compute a dictionnary, see :func:`boosting.boosting` for more information about boosting_param
        """
        m = data.shape[0]
        params = np.array([None]*m)
        for i in range(m) :
            X = data[i]
            y = X[1:,:]
            if print_step :
                print "data no "+str(i)

            (N,p) = y.shape

            obj = np.array([None] * self.maxiter)
            #obj[m].W is [p,p] interaction matrices, W[i,j] = 1 if i and j interact, 0 else
            #obj[m].rho is coeficient of the "line search" along the steepest-descent"
            #obj[m].h is base models learned from each subset
            Hm = np.tile(X.mean(),y.shape) #estimated boosting model, initialize by the average of gene expressions (data are centered)
            genes = np.array(range(p))
            nGenes = p
            stop = self.maxiter
            for m in range(self.maxiter ) :
                if print_step and (m/10)*10==m :
                    print "\t boosting step no "+str(m)
                #regularazion of h_m
                Um = y-Hm #Matrice of residuals
                if self.flagRes : #if the residual for gene i is too low then remove it
                    genesOut = []
                    for j in range(nGenes) :
                        if LA.norm(Um[:,genes[j]])**2 < self.eps :
                            genesOut.append(j)
                    genes[genesOut] = -1
                    genes = genes[genes>=0]
                if genes.size<=0 :
                    stop = m-1
                    print ('Stop at iteration_' + str(stop+1) + ': No more significant residuals')
                    break
                ##Interaction matrix learning
                terminate = True
                if (self.nFrac <= genes.size) :
                        nTry =0
                        while nTry<self.tot and terminate :
                            #select a random subset
                            idx_rand_m = genes.copy() # indices of genes selected at random
                            np.random.shuffle(idx_rand_m)
                            idx_rand_m = idx_rand_m[:self.nFrac]
                            idx_rand_m.sort()
                            (Wm_sub,terminate) = conditionalIndependence(Um[:,idx_rand_m], self.alpha, self.n_edge_pick)                            #partial correlation test
                            if terminate :  # if no significant edge was found in the subnetwork, choose another one
                                nTry = nTry+1
                            else :
                                Wm=np.zeros((p,p)) # TODO  : outside the loop ?
                                Wm[np.ix_(idx_rand_m,idx_rand_m)] = Wm_sub
                # if no significant edge was found after 'tot' trials
                if terminate :
                    stop = m-1
                    print ('Stop at iteration_' + str(stop+1) + ': Could not find significant edges')
                    break
                nGenes = genes.size # number of remaining genes
                L = np.diag(np.sum(Wm_sub,axis=0)) - Wm_sub
                B_m = sLA.expm(self.beta * L)
                self.kernel.B = B_m
                obj[m] = OKVAR.OKVAR(self.kernel,self.constraint)
                obj[m].beta = self.beta
                obj[m].fit(Um[:,idx_rand_m],True)
                #
                K_m = opera.kernels.gramMatrix(Um[:,idx_rand_m],Um[:,idx_rand_m],B_m,obj[m].kernel.gammadc,obj[m].kernel.gammatr)
                Z = K_m + self.constraint.lambda1*np.eye(K_m.shape[0])
                if LA.det(Z) == 0 :
                    stop = m-1
                    print ('Stop at iteration_' + str(stop+1) + ': Matrix K_m+lambda*Id is singular')
                    break
                else :
                    yNew = np.reshape(Um[:,idx_rand_m].T,(K_m.shape[0],1))
                    #(C_m_k,_,_) = proximal.elastic_shooting(K_m,yNew,obj.muH,obj.muC,init=np.linalg.solve(Z, yNew))
                    C_m_k = proximalLinear(K_m, yNew, Constraints=self.constraint, Loss=self.loss,maxiters=100)
                    if (C_m_k == 0).all() :
                        stop = m-1
                        print ('Stop at iteration_' + str(stop+1) + ': All regression coefficients are zero')
                        break
                obj[m].C = C_m_k
                obj[m].kernel.K = K_m
                #
                hm_k = np.reshape(np.dot(obj[m].kernel.matrix(),obj[m].C),(self.nFrac,N)).T
                hm = np.zeros((N,p)) # genes that do not belong to the subset are assigned a 0 prediction
                hm[:,idx_rand_m] = hm_k
                obj[m].h = hm
                obj[m].W = Wm
                Cm = np.zeros((p,N)) # rows of C_m = 0 for genes that do not belong to the subset
                Cm[idx_rand_m,:] = np.reshape(obj[m].C,(self.nFrac,N))
                obj[m].C = Cm
                obj[m].subset = idx_rand_m
                obj[m].rho = np.trace(np.dot(hm.T,Um))/LA.norm(hm,'fro')**2 #rho(m) = \arg\min_\rho ||D_m - \rho * h_m||_2^2;
                ##Update the boosting model
                Hm = Hm + obj[m].rho*hm
                obj[m].mse = (1./N)*((y-Hm)**2).sum();
            params[i]=obj[:stop]
        self.boost = params


    def predict_uc(self,data,jacobian_threshold=0.50,adj_matrix_threshold=0.50):
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
        params = self.boost
        for i in range(data.size) :
            Ji = np.abs(np.tanh(jacobian(self,data[i], params[i])))
            delta = quantile(vec(Ji),jacobian_threshold)
            M = M+(Ji>=delta)
        delta = quantile(vec(M),adj_matrix_threshold)
        M = (M>=delta)
        self.adj_matrix=M*1

        #TODO
        #Pour chaque data{i} :
        #    fit and predict <- 10fois
        #    add and threshold

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

