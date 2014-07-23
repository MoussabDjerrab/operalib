'''
.. module:: OVKR
   :platform: Unix, Windows
   :synopsis: module to performs an OVKR

.. moduleauthor:: Tristan Tchilinguirian <tristan.tchilinguirian@ensiie.fr>

Let there be :math:`N`-sized dataset :math:`\\chi`. We wish to estimate a :math:`p`-dimensional target function

.. math::
    h(x) = \\sum_{i=1}^N K\\left(x,x_i\\right) \\mathbf{c}_i \\in \\mathbb{R}
    
which is in vector notation

.. math::
    h(x) = \\mathbf{K}_{x,\\chi}\\overrightarrow{\\mathbf{c}} \\text{ where } \\mathbf{K}_{x,\\chi}\in\\mathbb{R}^{p\\times Np}

The full operator-valued kernel over the dataset :math:`\chi` is
   
    
.. math::

    \mathbf{K}_{\chi,\chi} = \\begin{pmatrix}
                             K(x_1,x_1)&         & \\cdots    &         & K(x_1,x_N) \\\\  
                                       & \\ddots &            &         &            \\\\
                             \\vdots   &         & K(x_i,x_j) &         & \\vdots    \\\\ 
                                       &         &            & \\ddots &            \\\\
                             K(x_N,x_1)&         & \\cdots    &         & K(x_N,x_N)
                            \\end{pmatrix}

where each block consists of :math:`p` times :math:`p` kernel values, for instance over the "components" :math:`i,j`

.. math::
    K(x,z) = \\begin{pmatrix}k(x^1,z^1)& \\cdots & k(x^1,z^j) & \\cdots & k(x^1,z^p) \\\\ \\vdots  & \\ddots && \\ddots &  \\\\ k(x^i,z^1)& & k(x^i,z^j) & & k(x^i,z^p) \\\\ \\vdots & \\ddots & & \\ddots & \\\\ k(x^p,z^1)& \\cdots & k(x^p,z^j) & \\cdots & k(x^p,z^p) \\end{pmatrix}

where :math:`p` is the number of targets. A common scalar kernel is the gaussian kernel

.. math::
    k(x,z) = \\exp\\left(-\\gamma ||x-z||^2\\right)

The OVK kernel method extends trivially the kernel ridge regression and classification.

'''

from .OPERAObject import OPERAObject
import numpy as np
import opera.kernels as kernels
from opera import proximal

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
    muC1s = [1]
    muC2s = [1]
    normCs = ["L1"]
    partitions = [None]
    weights = [None]

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
    if parameters.has_key('muC1') :
        muC1s = parameters['muC1']
    if parameters.has_key('muC2') :
        muC2s = parameters['muC2']
    if parameters.has_key('normC') :
        normCs = parameters['normC']
    if parameters.has_key('B') :
        Bs = parameters['B']
    if parameters.has_key('partition') :
        partitions = parameters['partition']
    if parameters.has_key('partition_weight') :
        weights = parameters['partition_weight']
        
    for ovkernel in ovkernels : 
        for kernel in kernels : 
            for c in cs : 
                for d in ds :
                    for gamma in gammas : 
                        for B in Bs : 
                            for muH in muHs : 
                                for muC1 in muC1s : 
                                    for normC in normCs :
                                        for muC2 in muC2s : 
                                            for partition in partitions : 
                                                for weight in weights : 
                                                    obj = OVKR(ovkernel,kernel,c,d,gamma,B,muH,normC,muC1,muC2,partition,weight)
                                        score = obj.crossvalidationscore(X, y, nblocks)
                                        if( score < bestscore) :
                                            bestmodel = obj
                                            bestscore = score
    return bestmodel

class OVKR(OPERAObject):
    """ 
    Performs OVK regression over parameter ranges, cross-validation, etc.
    
    :param ovkernel: 
    :type ovkernel: str, 'dc' or 'tr' 
    :param kernel:  
    :type kernel: str, 'linear' 'gauss' or 'polynomial' 
    :param B: 
    :type B: str - 'id' , 'cov' or nparray
    :param gamma: gaussian kernel gamma  
    :type gamma: float
    :param c: polynomial kernel c
    :type c: float
    :param d: polynomial kernel d
    :type d: float
    :param muH: regularizer for H
    :type muH: positif float
    :param muC: regularizer for C
    :type muC: positif float
    :param normC: norms for regularizers C
    :type normC: str, 'lasso' 'L2' 'elasticnet' 'grouplasso' 'sparsegrouplasso'
    """
    ovkernel = "dc"
    kernel = "gauss"
    c = 0
    d = 1
    gamma = 1
    B = "identity"
    muH = 1
    muC_1 = 1
    muC_2 = 1
    partitionC = None
    partitionC_weight=None
    normC = "L1"

    def __init__(self, ovkernel="dc",kernel="gauss",c=0,d=1,gamma=1,B="identity",muH=1,normC="L1",muC_1=1,muC_2=1,partitionC=None,partitionC_weight=None):
        self.ovkernel = ovkernel
        self.kernel = kernel 
        self.c = c
        self.d = d
        self.gamma = gamma
        self.B = B
        self.muH = muH
        self.normC = normC
        self.muC_1 = muC_1
        self.muC_2 = muC_2
        self.partitionC_weight=partitionC_weight
        self.partitionC=partitionC
        self.kernel_function = kernels.chooseFunctionKernel(ovkernel, kernel, c, d, gamma, B)

    def __repr__(self):
        if self.K is None : fitted = "no "
        else : fitted = "yes"
        return "OVKR model : < fitted:"+fitted+" >"
    def __str__(self):
        out = "OVKR model : \n\t Hyperparameters :\n"
        #parameters print
        def item(s,r):
            return "\t\t"+s+str(r)+"\n"
        out+=item("ridge penalty parameter : muH=",self.muH)
        out+=item("l1 penalty parameter : muC1=",self.muC1)
        out+=item("l2 penalty parameter : muC2=",self.muC2)
        if self.partitionC is not None : out+=item("partition of C : ",self.partitionC)
        if self.ovkernel == "tr" : ovkernel = "transformable"
        elif self.ovkernel == "dc" : ovkernel = "decomposable"
        else : ovkernel = str(self.ovkernel)
        if self.kernel == "linear" : kernel = "linear"
        elif self.kernel == "gauss" : kernel = "gaussian"
        elif self.kernel == "polynomial" : kernel = "polynomial"
        else : kernel = str(self.kernel)
        out+="The matrix_valued kernel is a "+ovkernel+kernel+" one"
        if self.B=="id" : out+="The matrix B is the [p,p] identity matrix"
        elif self.B=="cov" : out+="The matrix B is the [p,p] matrix, target covariance"
        elif self.B__class__==''.__class__ : out+="the matrix B is : "+self.B
        else : out+="The matrix B is already computed"
        out+=item("Parameter of gaussian matrix-valued kernel : gamma=",self.gamma)
        out+=item("Parameters of polynomial matrix-valued kernel : c="+str(self.c)+" d="+str(self.d),"")
        

        if self.n_edge_pick > 0 : item("Number of edges to pick in each random subset : n_edge_pick=",self.n_edge_pick)
        else :  item("Number of edges to pick in each random subset : n_edge_pick=","all the significant edges")
        if self.flagRes : f = "yes"
        else : f = "no"
        item("Variables whose residuals are too low are removed at each iteration : flagRes=",f)
        #fit print
        if self.K is None : out+="the model is not fitted yet"
        else : out+="the model is fitted"
        return out

    def learnC(self,Y):
        (N,_) = Y.shape
        Yvec = np.reshape(Y, (len(Y[0,:])*len(Y[:,0])))
        if (self.normC.lower() =='mixed' or self.normC.lower() == 'grouplasso'or self.normC.lower() == 'group lasso' or self.normC.lower() == 'sparsemixed' or self.normC.lower() == 'sparsegrouplasso'or self.normC.lower() == 'sparse group lasso' or self.normC.lower() == 'sparse mixed') and self.partitionC is None :
            partition = []
            for i in range(len(Y)/N) : 
                partition.append(np.array(range(N))+i*N)
            self.partitionC = np.array(partition)
            if self.partitionC_weight is None :
                self.partitionC_weight=np.ones(len(partition))
               
        Cvec = proximal.proximalLinear(self.K, Yvec, mu=self.muH, norm=self.normC, muX_1=self.muC_1, muX_2=self.muC_2 , partitionX = self.partitionC , partitionX_weight=self.partitionC_weight, eps=1.e-3)
        C = np.reshape(Cvec,Y.T.shape).T
        self.C = C
        return C

    
    def fit(self, X, y):
        """Method to fit a model
        
        :param: array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        :type: ndarray
        :param: array, with shape = [N,p], where N is the number of samples.
        :type: ndarray 
        """
        OPERAObject.fit(self, X, y)
        self.K = self.kernel_function(X,X,y)
        self.C = self.learnC(y)
        return
    
    def predict(self,X):
        """Method to predict theclust of a data
        
        :param: array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        :type: ndarray
        :returns: array, with shape = [N,p], where N is the number of samples.
        :rtype: ndarray 
        """
        Ktest = self.kernel_function(X,self.X,self.y)
        Cvec = np.reshape(self.C.T, (len(self.C[:,0])*len(self.C[0,:])))
        Yvec = np.dot(Ktest,Cvec)
        Y = np.reshape(Yvec,(len(Yvec)/len(self.C[0,:]),len(self.C[0,:])))
        return Y 

    def score(self,X,y):
        """Method to give a score of a model
        A model that can give a goodness of fit measure or a likelihood of unseen data, implements (higher is better):
        
        :param: array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        :type: ndarray
        :param: array, with shape = [N,p], where N is the number of samples.
        :type: ndarray 
        :returns: The score of our model
        :rtype: float
        """
        #compute the score
        ypred = self.predict(X)
        return np.mean((ypred - y)**2)
    
    def copy(self):
        return OVKR(ovkernel=self.ovkernel,kernel=self.kernel,c=self.c,d=self.d,gamma=self.gamma,B=self.B,muH=self.muH,muC_1=self.muC_1,muC_2=self.muC_2,partitionC=self.partitionC,normC=self.normC,partitionC_weight=self.partitionC_weight)
    
    def getparameter(self,show=True):
        if show :
            print   "ovkernel :\t %s\nkernel\t :\t %s\nc\t :\t %s\nd\t :\t %s\ngamma\t :\t %s\nB\t :\t %s\nmuH\t :\t %s\nnormC\t :\t %s\nmuC_1\t :\t %s\nmuC_2\t :\t %s\npartitionC\t :\t %s\nweigths group in partition\t :\t %s\n"% (self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B,self.muH,self.normC,self.muC_1,self.muC_2,self.partitionC,self.partitionC_weight)
        return [self.ovkernel,self.kernel,self.c,self.d,self.gamma,self.B,self.muH,self.normC,self.muC_1,self.muC_2,self.partitionC,self.partitionC_weight ]

    def setparam(self,name,val):
        if   name == "ovkernel" : self.ovkernel = val
        elif name == "kernel" : self.kernel = val
        elif name == "c" :self.c = val
        elif name == "d" :self.d = val
        elif name == "gamma" :self.gamma = val
        elif name == "B" : self.B = val
        elif name == "muH" : self.muH = val
        elif name == "muC1" : self.muC_1 = val
        elif name == "muC2" : self.muC_2 = val
        elif name == "partitionC" : self.partitionC = val
        elif name == "partitionCweights" : self.partitionC_weight = val
        elif name == "normC" : self.normC = val
        elif name == "kernel_function" : self.kernel_function = val
        return 