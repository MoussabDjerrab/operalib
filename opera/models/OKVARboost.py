'''
Created on Jun 18, 2014

@author: Tristan Tchilinguirian
'''

from .OPERAObject import OPERAObject
import numpy as np
import opera.kernels as kernels
from opera import proximal
import scipy.linalg as sLA
import numpy.linalg as LA
from opera.utils.conditionalIndependence import conditionalIndependence
from opera.utils import AUC
from scipy.stats.mstats import mquantiles as quantile
from opera.utils import vec

class OKVARboost(OPERAObject):
    """ 
    Performs OVK regression over parameter ranges, cross-validation, etc.
    
    Parameters 
        M :    number of boosting iterations
        gamma :    positive parameter of the scalar Gaussian kernel  
        muH :    (positive scalar) ridge penalty parameter
        muC :    (positive scalar) l1 penalty parameter
        randFrac :    size of random subset as a percentage of the network size
        alpha :    level of the partial correlation test
        n_edge_pick :    number of edges to pick in each random subset
                         n_edge_pick=0 means that the all the significant edges are picked
        eps :    stopping criterion threshold for residuals
        flagRes :    if not 0, variables whose residuals are too low are
                     removed at each iteration 

    Methods : 
        fit : X,y -> fit a model
        predict : X* -> y* the predicted classes
        score : X,y -> score of the model with X and y
        crossvalidation_score : X,y,B -> give a crossvalidation error of the model with B bloc
        copy : self -> another model with the same parameters ans methods
        setparameters : val_name,val -> assign val at val_name
        getparameters : bool -> give the parameters, if bool it's true print them
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
        '''
        Constructor
        '''
        self.gammadc = gammadc
        self.gammatr = gammatr
        
        self.muH = muH
        self.muC = muC
        self.randFrac = randFrac
        self.alpha = alpha
        self.n_edge_pick = n_edge_pick
        self.eps = eps
        self.flagRes = flagRes
        self.max_iter = max_iter
        self.boosting_param = None
        self.adj_matrix = None

    def fit(self,data):
        """Method to fit a model
        Parameter
            data : cell of N [array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
        """
        N = data.size
        params = np.array([None]*N)
        for i in range(N) : 
            params[i] = self.boosting(data[i])
        self.boosting_param = params

    def predict(self,data,jacobian_threshold=1,adj_matrix_threshold=0.50):
        """Method to predict a model
        Parameter
            data : cell of N [array-like, with shape = [n,d], where n is the number of samples and d is the number of features.
            jacobian_threshold : quantile level of the Jacobian values used to get the adjacency matrix
            adj_matrix_threshold : M = sum{for each cell i}(Mi). our final matrix keeps the pourcentage of our adj_matrix_threshold ceil
        """
        (_,p) = data[0].shape
        M = np.zeros((p,p))
        params = self.boosting_param
        for i in range(data.size) :
            Ji = self.jacobian(data[i], params[i])
            delta = quantile(vec(np.abs(np.tanh(Ji))),jacobian_threshold)
            M = M+(Ji>=delta)
            
        #on conserve les adj_matrix_threshold meilleurs elements non nuls
        A = (M.reshape(M.size)).copy()
        A.sort()
        n_keep = ((A>0).sum()*adj_matrix_threshold)
        if n_keep<=0 : 
            M = (M>0)
        else :
            M = (M>=A[A.size-n_keep])
        #M = (M>=(adj_matrix_threshold*M.sum()))
        self.adj_matrix=M*1

    def score(self, data, M, jacobian_threshold=1,adj_matrix_threshold=0.50):
        self.predict(data, jacobian_threshold, adj_matrix_threshold)
        M_vec = np.reshape(self.adj_matrix,self.adj_matrix.size)
        Mvec = np.reshape(M,M.size)
        return AUC(Mvec,M_vec)

    def boosting(self, X,y=None):
        """Method to do a boosting on a model
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N,p], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        Output (dictionnary) 
            W
            C
            rho
            subset
            h
            J
        """
        output = {
              "W"   : None ,
              "C"   : None ,
              "rho" : None , 
              "subset" : None ,
              "h"   : None , 
              "J"   : None ,
              "nb_iter" : None
              }
        if y is None : 
            y=X[1:,:]
        (N,p) = y.shape
        # M is the number of boosting iterations
        M = self.max_iter 
        # Ws is a list of W : [p,p] interaction matrices W_m
        #      W_m[i,j] = 1 -> genes i and j interact
        #      W_m[i,j] = 0 -> gene i and j do not interact
        W = np.array([None] * M)
        # Cs is a list of C : Cm parameters
        C = np.array([None] * M)
        #rhos is the coeficient of the "line search" along the steepest-descent"
        rho = np.zeros((M,1))
        #hs is a list of h : base models learned from each subset
        h = np.array([None] * M)
        #H_m is a matrix : estimated boosting model, initialize by the average of gene expressions (data are centered)
        H_m = np.tile(y.mean(),y.shape)
        
        #Mean Squared Errors
        mse = np.zeros((M,p))
        #our subsets : a list of subset
        subset = np.array([None] * M)
        
        #size of random subsets
        nFrac = np.round(self.randFrac*p)
        
        genes = np.array(range(p))
        nGenes = p
        
        
        #TODO : see if it has to be arguments
        #some other stuff
        tot = 1000 #maximum number of trials for the partial correlation test
        betaParam = 0.2 # diffusion parameter for the Laplacian
        
        stop = M #stopping iteration
        
        for m in range(M) : 
            #regularazion of h_m
            #Matrice of residuals
            U_m = y-H_m
            #if the residual for gene i is too low then remove it
            if self.flagRes : 
                genesOut = []
                for j in range(nGenes) : 
                    if LA.norm(U_m[:,genes[j]])**2 < self.eps : 
                        genesOut.append(j)
                genes[genesOut] = -1
                genes = genes[genes>=0]
            #
            if genes.size<=0 :  
                stop = m-1
                print ('Stop at iteration_' + str(stop+1) + ': No more significant residuals')
                break
            #
            ##Interaction matrix learning 
            terminate = True
            if (nFrac <= genes.size) : 
                    nTry =0 
                    while nTry<tot and terminate : 
                        #
                        #select a random subset
                        idx_rand_m = genes.copy() # indices of genes selected at random
                        np.random.shuffle(idx_rand_m)
                        idx_rand_m = idx_rand_m[:nFrac]
                        idx_rand_m.sort()
                        #
                        #partial correlation test
                        (W_m_sub,terminate) = conditionalIndependence(U_m[:,idx_rand_m], self.alpha, self.n_edge_pick)
                        #
                        # if no significant edge was found in the subnetwork, choose another one
                        if terminate :
                            nTry = nTry+1
                        else : 
                            W_m=np.zeros((p,p)) # TODO  : outside the loop ?
                            W_m[np.ix_(idx_rand_m,idx_rand_m)] = W_m_sub 
            # if no significant edge was found after 'tot' trials
            if terminate : 
                stop = m-1
                print ('Stop at iteration_' + str(stop+1) + ': Could not find significant edges')
                break
            #
            # number of remaining genes
            nGenes = genes.size
            #
            ## Gram Matrix computation
            # Laplacian
            L = np.diag(np.sum(W_m_sub,axis=0)) - W_m_sub
            B_m = sLA.expm(betaParam * L)
            #Gram Matrix
            K_m = (kernels.gramMatrix(U_m[:,idx_rand_m],U_m[:,idx_rand_m],B_m,self.gammadc,self.gammatr))
            #
            #
            ## Coefficient Cs learning
            Z = K_m + self.muH*np.eye(K_m.shape[0])
            if LA.det(Z) == 0 : 
                stop = m-1
                print ('Stop at iteration_' + str(stop+1) + ': Matrix K_m+lambda*Id is singular')
                break
            else : 
                yNew = np.reshape(U_m[:,idx_rand_m].T,(K_m.shape[0],1))
                C_m_k = proximal.proximalLinear(K=K_m, y=yNew, mu=self.muH, norm='l1', muX_1=self.muC,maxiters=10, eps=0)
                if (C_m_k == 0).all() : 
                    stop = m-1
                    print ('Stop at iteration_' + str(stop+1) + ': All regression coefficients are zero')
                    break
            #
            h_m_k = np.reshape(np.dot(K_m,C_m_k),(nFrac,N)).T
            h_m = np.zeros((N,p)) # genes that do not belong to the subset are assigned a 0 prediction
            h_m[:,idx_rand_m] = h_m_k
            h[m] = h_m
            #
            W[m] = W_m
            #
            C_m_k = np.reshape(C_m_k,(nFrac,N))
            C_m = np.zeros((p,N)) # rows of C_m = 0 for genes that do not belong to the subset
            C_m[idx_rand_m,:] = C_m_k
            C[m] = C_m
            #
            subset[m] = idx_rand_m
            #
            ## Line search
            rho[m] = np.trace(np.dot(h_m.T,U_m))/LA.norm(h_m,'fro')**2 #rho(m) = \arg\min_\rho ||D_m - \rho * h_m||_2^2;
            ##Update the boosting model
            H_m = H_m + rho[m]*h_m
            #print("K_m : \n\t loop  "+str(m)+"\n\t min   "+str(K_m.min())+"\n\t max   "+str(K_m.max())+"\n\t mean  "+str(K_m.mean())+"\n|C_m|inf "+str(LA.norm(C_m_k,float("inf")))+"\n|H_m|inf "+str(LA.norm(H_m,float("inf"))))
            #
            ## Compute the Mean Squared Errors
            mse[m,:] = (1/N)*((y-H_m)**2).sum();
            
        ##Resize the outputs   
        output["W"] = W[:stop]
        output["C"] = C[:stop]
        output["rho"] = rho[:stop]
        output["subset"] = subset[:stop]
        output["h"] = h[:stop]
        output["nb_iter"] = stop
        return output
    
    def jacobian(self,data,param):
        """Method to compute a jacobian on a model after a boosting
        
        Parameters      
            data        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            param    the output of a boosting
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
            #
            tmpJ_sub = np.zeros((nFrac,nFrac))
            tmpJ = np.zeros((p,p))
            # iterate over time points 1:tf
            for t in range(tf):
                for l in range(tf) :
                    K_lt = (kernels.trgauss(data[l,subset[m]].T,data[t,subset[m]].T,self.gammatr))
                    tmpJ1 = np.tile(data[l,subset[m]],(nFrac,1))-np.tile(data[t,subset[m]],(nFrac,1))
                    tmpJ2 = np.tile(C_m_sub[:,l],(nFrac,1))
                    tmpJ_sub = tmpJ_sub + tmpJ1*K_lt*tmpJ2
    
            tmpJ[np.ix_(subset[m],subset[m])] = tmpJ_sub
            B_m = np.zeros((p,p));
            B_m[np.ix_(subset[m],subset[m])] = B_m_sub;
            #
            J = J + (2./tf)*self.gammatr*rho[m]*B_m*tmpJ
            Js[m] = J 
        return J