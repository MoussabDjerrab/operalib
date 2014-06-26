import abc
from opera import kernels, proximal
from opera.models.split_data import __init__ as split_data
import numpy as np

class OPERAObject(object):
    __metaclass__ = abc.ABCMeta
    X = None
    y = None
    K = None
    C = None

    @abc.abstractmethod
    def fit(self, X, y, kwargs):
        """Method to fit a model
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        """
        self.X=X
        self.y=y
        return self
    def predict(self, X):
        """Method to predict theclust of a data
        
        Parameters      
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        Output
            y        array, with shape = [N], where N is the number of samples.
        """
        # compute predictions ''pred''
        pred = None
        return pred
    def transformer(self, X):
        """Method to transform data
        
        Transforms the input data - e.g. selects a subset of the features or extracts new features based on the original ones
        """
        #compute ''X'' to ''X_prime''
        X_prime = None
        return X_prime
    def score(self,X,y):
        """Method to give a score of a model
        A model that can give a goodness of fit measure or a likelihood of unseen data, implements (higher is better):
        """
        #compute the score
        gof = 0
        return gof
    

    def crossvalidationscore(self,X,y,B=5) : 
        blocs = np.array(split_data(X, y, 1, B-1))
        score = []
        obj = self.copy()
        
        for i in range(blocs.shape[0]) :
            # test data bloc is the ith ans train data blocs are the others  
            L = blocs.tolist()
            xtest,ytest = L.pop(i)
            xtest = np.array(xtest)
            ytest = np.array(ytest)
            xtrain,ytrain = L.pop()
            while L != [] :
                x,y = L.pop()
                xtrain = np.concatenate((xtrain,x))
                ytrain = np.concatenate((ytrain,y))
                
            obj.fit(xtrain,ytrain)
            score.append(obj.score(xtest, ytest))

        return np.array(score).mean()
    
    def copy(self):
        return None
    
    def getparameter(self):
        return[]
    @classmethod
    
    def learnC(self,K=K,Y=y,muH=1,muC=1,normC="L1"):
        
        
        Yvec = np.reshape(Y, (len(Y[0,:])*len(Y[:,0])))
        Cvec = proximal.proximalLinear(K, Yvec, mu=muH, muX=muC, norm=normC, n=Y.shape[0])
        C = np.reshape(Cvec,Y.T.shape).T
        self.C = C
        return C
    
    def computeKernel(self,ovkernel="dc",kernel="gauss",c=1,d=3,gamma=1,B="identity",Xtest=None):
        """ Compute a kernel
        ovkernel is : 
            dc : transformable : K(x,z)_(ij) = k(xi,zi)
            tf : decomposable : K(x,z) = B x k(x,z)
            custom : the user give a K
        kernel is : 
            gauss : gaussian kernel
            linear : linear kernel
            poly : polunomial kernel
            custom : the user give a k
        other parameter : 
            c and d for linear kernel (default c=1 and d=3) 
            g for gaussian kernel (default g=1)
            B for decomposable type (default B='identity')
                identity : B is id
                cov : B is cov(Y)
                learn : the algo learn B
                custom : the user give a B
        """
        #if we compute K to predict a y then we do not chance self.K
        stock_K=False
        if Xtest == None : 
            Xtest = self.X.copy()
            stock_K=True
        
        if B == "identity" or B == None : 
            B = np.identity(len(self.y[0,:]))
        elif B == "cov" :
            B = np.cov(self.y)
        #elif B == "learn" : 
            #B = kernels.learn()
            
        if ovkernel=="dc" or ovkernel == None :
            if kernel == "gauss" or kernel == None :
                K = kernels.dcgauss(Xtest,self.X,gamma,B)
            elif kernel == "linear" : 
                K = kernels.dclin(Xtest,self.X,B)
            elif kernel == "polynomial" : 
                K = kernels.dclin(Xtest,self.X,c,d,B)
            elif kernel.__class__ == np.ndarray :
                K = kernels.dccust(kernel)
        elif ovkernel == "tr" : 
            if kernel == "gauss"  or kernel == None :
                K = kernels.trgauss()
            elif kernel == "linear" : 
                K = kernels.trlin()
            elif kernel.__class__ == np.ndarray :
                K = kernels.trcust(kernel)
        elif ovkernel.__class__ == np.ndarray : 
            K = ovkernel.copy()
        
        if(stock_K):
            self.K=K
        
        return K

    