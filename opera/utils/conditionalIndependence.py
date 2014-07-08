import numpy as np
import numpy.linalg as LA
from scipy.stats import norm

def conditionalIndependence(data,alpha,n_edge_pick=0):
    """
    Test conditional independence between variables using partial correlations
    
    INPUTS :
        data :    ([N,p] matrix) N is the number of p-dimensional observations
        alpha :    significance level of the test
        NE_pick :    number of edges to pick in each random subset
                     NE_pick=0 means that the all the significant edges are picked
    OUTPUTS :
        ([p,p] matrix) Res(i,j) = 1 if the partial correlation between X_i and X_j given the other variables is significant
    """
    
    def cov(x,y): return ((len(x)-1.)/len(x)*np.cov(x,y))[0,1]
    
    (N,p) = data.shape
    covMat = np.zeros((p,p))
    Res = np.zeros((p,p))
    
    #compute the covariance matrix with time lag 1
    for i in range(p) : 
        for j in range(i+1) : 
            cov_ij = cov(data[0:N-1,i],data[1:N,j]) 
            cov_ji = cov(data[0:N-1,j],data[1:N,i])  
            if np.abs(cov_ij) >= np.abs(cov_ji) :
                covMat[i,j] = cov_ij
            else : 
                covMat[i,j] = cov_ji
            covMat[j,i] = covMat[i,j]

        
    iscovMatSingular = False

    if (LA.det(covMat) == 0) : 
        iscovMatSingular = True
    else : 
        precisionMat = LA.inv(covMat)*(1+0j) 
        invdiagpMat = np.array([1/np.sqrt(np.diag(precisionMat))])
        partialCorrMat = - np.tile(invdiagpMat.T,(1,p)) * precisionMat *  np.tile(invdiagpMat,(p,1));
        Stat = np.abs(np.arctanh(partialCorrMat)) * np.sqrt(N-(p-2)-3) # test statistic
        Test = Stat > norm.ppf(1-0.5*alpha,0,1);
        
        if n_edge_pick == 0 : 
            Res = Test;
        else : 
            SigStat = Stat * Test # significant statistic values
            ind = np.reshape(np.array(range(p*p)),(p,p));
            idxEdge = ind.T[np.tril(np.ones(ind.shape),-1)>0];
            sigStat = SigStat[np.tril(SigStat,-1)!=0];
            order = sigStat.argsort()
            idxSigEdge = idxEdge[order]; # significant edges
            # If the network is sparse and can't choose m edges, then choose |sigEdge|
            last_m = min((n_edge_pick,len(idxSigEdge)));
            if(n_edge_pick > last_m) :
                Warning('Chose_'+last_m+'_edges instead of the requested_'+n_edge_pick);
            idxPickedEdge = idxSigEdge[np.array(range(last_m))]
            Res[idxPickedEdge] = 1;
    return (Res,iscovMatSingular)# = symmetric(Res);

    