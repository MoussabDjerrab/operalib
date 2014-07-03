'''
Created on Jun 18, 2014

@author: Tristan Tchilinguirian
'''

from .OPERAObject import OPERAObject
import numpy as np
import opera.kernels as kernels

def grid_search(X,y,nblocks=5,parameters={}):
    """
    Do a search of the best choices of parameter by minimizing the crossvalidation score with nblocks blocks
    """
    
    bestscore = float("inf")
    bestmodel = None
    gammas = [1]
    cs = [1]
    ds = [1]
    ovkernels = ["dc"]
    kernels = ["gauss"]
    Bs = ["identity"]
    muHs = [1]
    muCs = [1]
    normCs = ["L1"]

    if parameters.has_key('gamma') : 
        gammas = parameters['gamma'] 
    if parameters.has_key('c') : 
        cs = parameters['c']
    if parameters.has_key('d') :
        ds = parameters['d']
    if parameters.has_key('ovkernel') :
        ovkernels = parameters['ovkernel']
    if parameters.has_key('kernel') :
        kernels = parameters['kernel']
    if parameters.has_key('muH') :
        muHs = parameters['muH']
    if parameters.has_key('muC') :
        muCs = parameters['muC']
    if parameters.has_key('normC') :
        normCs = parameters['normC']
    if parameters.has_key('B') :
        Bs = parameters['B']
        
    for ovkernel in ovkernels : 
        for kernel in kernels : 
            for c in cs : 
                for d in ds :
                    for gamma in gammas : 
                        for B in Bs : 
                            for muH in muHs : 
                                for muC in muCs : 
                                    for normC in normCs :
                                        obj = OVKR(ovkernel, kernel, c, d, gamma, B, muH, muC, normC)
                                        score = obj.crossvalidationscore(X, y, nblocks)
                                        if( score < bestscore) :
                                            bestmodel = obj
                                            bestscore = score
    return bestmodel

class OVKR(OPERAObject):
    """ 
    Performs OVK regression over parameter ranges, cross-validation, etc.
    
    Parameters
        ovkernel : 
            dc : decomposable
            tr : transformable
        kernel: 
            linear : linear kernel
            gauss : gaussian kernel
            polynomial : polynomial kernel
        B:
            id : [p,p] identity matrix
            cov : [p,p] matrix, target 'covariance'
        gamma:    gaussian kernel gamma
        c :    polynomial kernel c
        d :    polynomial kernel d
        muH : regularizer for H
        muC : regularizer for C
        normC : norms for regularizers C
            L1
            mixed

    Methods : 
        fit : X,y -> fit a model
        predict : X* -> y* the predicted classes
        score : X,y -> score of the model with X and y
        crossvalidation_score : X,y,B -> give a crossvalidation error of the model with B bloc
        copy : self -> another model with the same parameters ans methods
        setparameters : val_name,val -> assign val at val_name
        getparameters : bool -> give the parameters, if bool it's true print them
    """
    ovkernel = "dc"
    kernel = "gauss"
    c = 1
    d = 1
    gamma = 1
    B = "identity"
    muH = 1
    muC_1 = 1
    muC_2 = 1
    partitionC = None
    partitionC_weight=None
    normC = "L1"

    def __init__(self, ovkernel="dc",kernel="gauss",c=1,d=1,gamma=1,B="identity",normC="L1",muH=1,muC_1=1,muC_2=1,partitionC=None,partitionC_weight=None):
        '''
        Constructor
        '''
        self.ovkernel = ovkernel
        self.kernel = kernel 
        self.c = c
        self.d = d
        self.gamma = gamma
        self.B = B
        self.muH = muH
        self.muC_1 = muC_1
        self.muC_2 = muC_2
        self.partitionC_weight=partitionC_weight
        self.partitionC=partitionC
        self.normC = normC
        self.kernel_function = kernels.chooseFunctionKernel(ovkernel, kernel, c, d, gamma, B)

    
    def fit(self, X, y, kwargs=None):
        """Method to fit a model
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N,p], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        """
        OPERAObject.fit(self, X, y, kwargs)
        self.K = self.kernel_function(X,X,y)
        self.learnC(K=self.K,Y=self.y,muH=self.muH,muC=self.muC,normC=self.normC)
        return
    
    def predict(self,X):
        """Method to predict theclust of a data
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        Output
            y        array, with shape = [N,p], where N is the number of samples.
        """
        Ktest = self.kernel_function(X,self.X,self.y)
        Cvec = np.reshape(self.C.T, (len(self.C[:,0])*len(self.C[0,:])))
        Yvec = np.dot(Ktest,Cvec)
        Y = np.reshape(Yvec,(len(Yvec)/len(self.C[0,:]),len(self.C[0,:])))
        return Y 

    def score(self,X,y):
        """Method to give a score of a model
        A model that can give a goodness of fit measure or a likelihood of unseen data, implements (higher is better):
        """
        #compute the score
        ypred = self.predict(X)
        return np.mean((ypred - y)**2)
    
    def copy(self):
        return OVKR(ovkernel=self.ovkernel,kernel=self.kernel,c=self.c,d=self.d,gamma=self.gamma,B=self.B,muH=self.muH,muC=self.muC,normC=self.normC)
    
    def getparameter(self,show=True):
        if show :
            print   "ovkernel :\t %s\nkernel\t :\t %s\nc\t :\t %s\nd\t :\t %s\ngamma\t :\t %s\nB\t :\t %s\nmuH\t :\t %s\nmuC\t :\t %s\nnormC\t :\t %s\n"% (self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B,self.muH,self.muC,self.normC)
        return [self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B,self.muH,self.muC,self.normC ]

    def setparam(self,name,val):
        if   name == "ovkernel" : self.ovkernel = val
        elif name == "kernel" : self.kernel = val
        elif name == "c" :self.c = val
        elif name == "d" :self.d = val
        elif name == "gamma" :self.gamma = val
        elif name == "B" : self.B = val
        elif name == "muH" : self.muH = val
        elif name == "muC" : self.muC = val
        elif name == "normC" : self.normC = val
        elif name == "kernel_function" : self.kernel_function = val
        return 