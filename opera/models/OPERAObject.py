import abc
from opera import proximal
from opera.models.split_data import __init__ as split_data
import numpy as np

class OPERAObject(object):
    __metaclass__ = abc.ABCMeta
    X = None
    y = None
    K = None
    C = None
    kernel_function = None

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
    