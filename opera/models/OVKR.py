'''
Created on Jun 18, 2014

@author: Tristan Tchilinguirian
'''

from .OPERAObject import OPERAObject
import numpy as np


def grid_search(X,y,nbloc=5,parameters={}):
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
                                        score = obj.crossvalidationscore(X, y, nbloc)
                                        if( score < bestscore) :
                                            bestmodel = obj
                                            bestscore = score
    return bestmodel

class OVKR(OPERAObject):
    '''
    classdocs
    '''
    ovkernel = "dc"
    kernel = "gauss"
    c = 1
    d = 1
    gamma = 1
    B = "identity"
    muH = 1
    muC = 1
    normC = "L1"

    def __init__(self, ovkernel="dc",kernel="gauss",c=1,d=1,gamma=1,B="identity",muH=1,muC=1,normC="L1"):
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
        self.muC = muC
        self.normC = normC
    
    def fit(self, X, y, kwargs=None):
        """Method to fit a model
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N,p], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        """
        OPERAObject.fit(self, X, y, kwargs)
        self.computeKernel(self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B)
        self.learnC(K=self.K,Y=self.y,muH=self.muH,muC=self.muC,normC=self.normC)
        return
    
    def predict(self,X):
        """Method to predict theclust of a data
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        Output
            y        array, with shape = [N,p], where N is the number of samples.
        """
        Ktest = self.computeKernel(self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B,Xtest=X)
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
    
    def getparameter(self):
        return ["ovkernel","kernel","c","d","gamma","B","muH","muC","normC"]
        
