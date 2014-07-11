'''
Created on Jun 18, 2014

@author: Tristan Tchilinguirian
'''

from .OPERAObject import OPERAObject
import numpy as np
import opera.kernels as kernels
from opera import proximal
import numpy.linalg as LA
from opera.utils.conditionalIndependence import conditionalIndependence

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

    def __init__(self,gammadc=1e-4,gammatr=1,muH=0.001,muC=1,randFrac=1,alpha=0.05,n_edge_pick=0,eps=1e-4,flagRes=1,max_iter=100):
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
        self.eps = self.eps
        self.flagRes = self.flagRes


    def fit(self, X, y, kwargs=None):
        """Method to fit a model
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N,p], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        """
        OPERAObject.fit(self, X, y, kwargs)
        
        (N,p) = y.shape
        # M is the number of boosting iterations
        M = self.max_iter 
        # Ws is a list of W : [p,p] interaction matrices W_m
        #      W_m[i,j] = 1 -> genes i and j interact
        #      W_m[i,j] = 0 -> gene i and j do not interact
        Ws = np.array([None] * M)
        # Cs is a list of C : Cm parameters
        C = np.array([None] * M)
        #rhos is the coeficient of the "line search" along the steepest-descent"
        rhos = np.zeros((M,1))
        #hs is a list of h : base models learned from each subset
        hs = np.array([None] * M)
        #H_m is a matrix : estimated boosting model, initialize by the average of gene expressions (data are centered)
        H_m = np.tile(y.mean,(N,1))
        
        #Mean Squared Errors
        mse = np.zeros((M,p))
        #our subsets : a list of subset
        subsets = np.array([None] * M)
        
        #size of random subsets
        nFrac = np.round(self.randFrac*p)
        
        genes = np.array(range(p))
        nGenes = p
        
        
        #TODO : see if it has to be arguments
        #some other stuff
        tot = 100 #maximum number of trials for the partial correlation test
        betaParam = 0.2 # diffusion parameter for the Laplacian
        
        stop = M #stopping iteration
        
        for m in range(M) : 
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
                        #
                        #partial correlation test
                        (W_m_sub,terminate) = conditionalIndependence(U_m[:,idx_rand_m], self.alpha, self.n_edge_pick)
                        #
                        # if no significant edge was found in the subnetwork, choose another one
                        if terminate :
                            nTry = nTry+1
                        else : 
                            W_m=np.zeros((p,p)) # TODO  : outside the loop ?
                            W_m[idx_rand_m,idx_rand_m] = W_m_sub 
            # if no significant edge was found after 'tot' trials
            if terminate : 
                stop = m-1
                break
            #
            # number of remaining genes
            nGenes = genes.size
            #
            ## Gram Matrix computation
            # Laplacian
            L = np.diag(np.sum(W_m_sub)) - W_m_sub
            B_m = np.exp(betaParam * L)
            #Gram Matrix
            K_m = kernels.gramMatrix(U_m[:,idx_rand_m],U_m[:,idx_rand_m],B_m,self.gammadc,self.gammatr)
            #
            ## Coefficient Cs learning
            Z = K_m + self.muH*np.eye(K_m.shape)
            if LA.det(Z) == 0 : 
                stop = m-1
                break
            else : 
                yNew = np.reshape(U_m[:,idx_rand_m].T,(K_m.shape[0],1))
                C_m_k = proximal.proximalLinear(K=K_m, y=yNew, mu=self.muH, norm='l1', muX_1=self.muC)
                if (C_m_k == 0).all() : 
                    stop = m-1
                    break
            #
            h_m_k = np.reshape(np.dot(K_m,C_m_k),(nFrac,N)).T
            h_m = np.zeros((N,p)) # genes that do not belong to the subset are assigned a 0 prediction
            h_m[:,idx_rand_m] = h_m_k
            hs[m] = h_m
            #
            Ws[m] = W_m
            #
            C_m_k = np.reshape(C_m_k,(nFrac,N))
            C_m = np.zeros((p,N)) # rows of C_m = 0 for genes that do not belong to the subset
            C_m[idx_rand_m,:] = C_m_k
            C[m] = C_m
            #
            subsets[m] = idx_rand_m
            #
            ## Line search
            rhos[m] = np.trace(np.dot(h_m.T,U_m))*LA.norm(h_m,'fro')**2 #rho(m) = \arg\min_\rho ||D_m - \rho * h_m||_2^2;
            ##Update the boosting model
            H_m = H_m + np.dot(rhos[m],h_m)
            ## Compute the Mean Squared Errors
            mse[m,:] = (1/N)*((y-H_m)**2).sum();
            #
            ##Resize the outputs
            Ws = Ws[:stop]
            C = C[:stop]
            rhos = rhos[:stop]
            mse = mse[:stop,:]
            subsets = subsets[:stop]
            hs = hs[:stop]

        return(Ws,C,rhos,hs,H_m)