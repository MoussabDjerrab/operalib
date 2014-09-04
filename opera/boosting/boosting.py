import numpy as np
import opera.kernels as kernels
from opera.proximal import proximalLinear as prox
from opera.proximal import elastic_shooting
import scipy.linalg as sLA
import numpy.linalg as LA
from opera.utils.conditionalIndependence import conditionalIndependence
from opera.models.OKVAR import OKVAR


def boosting(obj,X,y=None,print_step=False):
    """Method to do a boosting on a model

    Parameters
    ----------
        X : ndarray [n,d] where n is the number of samples and d is the number of features.

        y : optional, ndarray [n,d]
            if y is None then boosting(X) is equivalent to boosting(X[:n-1],X[1:])

        print_step : bool (default=False)
            If it is true then displayed to the user the current step on the standard output

    Result
    ------
        params : dictionnary, each arguments is a list because each argument of each list corresponds to an iteration of the boosting
            W : nparray
                list of [p,p] interaction matrices W_m
                    W_m[i,j] = 1 -> genes i and j interact
                    W_m[i,j] = 0 -> gene i and j do not interact
            C : nparray
                list of Cm parameters
            rho : nparray
                list of the coeficient of the "line search" along the "steepest-descent"
            h : nparray
                list of h : base models learned from each subset
            subset : nparray
                list of subset (nparray)
            J : nparray
                list of
            mse : nparray
                list of mean squared errors
    """
    if y is None :
        y=X[1:,:]
    (N,p) = y.shape
    # M is the number of boosting iterations
    M = obj.maxiter
    param = np.array([None] * M)
    #  param[m].W [p,p] interaction matrices W_m; [i,j] is True if i and j interact
    #  param[m].C C parameters
    #  param[m].rho is the coeficient of the "line search" along the steepest-descent"
    #  param[m].h is a list of h : base models learned from each subset
    #  param[m].mse is the Mean Squared Errors
    #  param[m].subset is the subset used at iteration m

    #H_m is a matrix : estimated boosting model, initialize by the average of gene expressions (data are centered)
    H_m = np.tile(X.mean(),y.shape)
    #size of random subsets
    nFrac = np.round(obj.nFrac*p)
    #Our genes (it will evolve)
    genes = np.array(range(p))
    nGenes = p
    #TODO : see if it has to be arguments
    #some other stuff
    tot = obj.tot #maximum number of trials for the partial correlation test
    betaParam = obj.beta # diffusion parameter for the Laplacian

    stop = M #stopping iteration

    for m in range(M) :
        if print_step and (m/10)*10==m :
            print "\t boosting step no "+str(m)
        #regularazion of h_m
        #Matrice of residuals
        U_m = y-H_m
        #if the residual for gene i is too low then remove it
        if obj.flagRes :
            genesOut = []
            for j in range(nGenes) :
                if LA.norm(U_m[:,genes[j]])**2 < obj.eps :
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
                    (W_m_sub,terminate) = conditionalIndependence(U_m[:,idx_rand_m], obj.alpha, obj.n_edge_pick)
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
        obj.kernel.B = B_m
        #
        param[m] = OKVAR(obj.kernel,obj.constraint)
        param[m].beta = obj.beta
        # TODO corect (added some words)
        param[m].fit(U_m[:,idx_rand_m],True)
        param[m].kernel.K = kernels.gramMatrix(U_m[:,idx_rand_m],U_m[:,idx_rand_m],B_m,obj.kernel.gammadc,obj.kernel.gammatr)
        param[m].C = prox(K=param[m].kernel.K, y=np.reshape(U_m[:,idx_rand_m].T,(param[m].kernel.K.shape[0],1)), Constraints=obj.constraint,maxiters=100)
        h_m_k = np.reshape(np.dot(param[m].kernel.matrix(),param[m].C),(nFrac,N)).T
        h_m = np.zeros((N,p)) # genes that do not belong to the subset are assigned a 0 prediction
        h_m[:,idx_rand_m] = h_m_k
        param[m].W = W_m
        C_m_k = np.reshape(param[m].C,(nFrac,N))
        param[m].C = np.zeros((p,N)) # rows of C_m = 0 for genes that do not belong to the subset
        param[m].C[idx_rand_m,:] = C_m_k
        param[m].subset = idx_rand_m
        ## Line search
        param[m].rho = np.trace(np.dot(h_m.T,U_m))/LA.norm(h_m,'fro')**2
        ##Update the boosting model
        H_m = H_m + param[m].rho*h_m
        #print("K_m : \n\t loop  "+str(m)+"\n\t min   "+str(K_m.min())+"\n\t max   "+str(K_m.max())+"\n\t mean  "+str(K_m.mean())+"\n|C_m|inf "+str(LA.norm(C_m_k,float("inf")))+"\n|H_m|inf "+str(LA.norm(H_m,float("inf"))))
        ## Compute the Mean Squared Errors
        param[m].mse = (1./N)*((y-H_m)**2).sum();
    ##Resize the output
    return param[:stop]

