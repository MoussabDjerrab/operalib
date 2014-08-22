import abc
from opera.models.split_data import __init__ as split_data
import numpy as np

class OPERAObject(object):
    #__metaclass__ = abc.ABCMeta
    def __init__(self,kernel,constraint,loss):
        self.kernel = kernel
        self.constraint = constraint
        self.loss = loss
    @abc.abstractmethod
    def fit(self, X, y, kwargs=None):
        """Method to fit a model

        Parameters
            X        array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
            y        array, with shape = [N], where N is the number of samples.
            kwargs    optional data-dependent parameters.
        """
        self.datas=X
        self.labels=y
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
        """Method to give a cross-validation score of a model
        B is the number of bloc we have when we split the data
        """
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
        """ copy a model, here it is nothing but it have to be define in each subclass"""
        return OPERAObject(self.kernel.copy(),self.constraint.copy(),self.loss.copy())


    def getparameter(self):
        """ give the parameters of a model, here it is nothing but it have to be define in each subclass"""
        return { "kernel" : self.kernel ,
                 "constraint" : self.constraint ,
                 "loss" : self.loss
                }

    def setparam(self,name,val):
        n = name.lower
        if n == "kernel" : self.kernel = val
        elif n == "constraint" : self.constraint = val
        elif n == "loss" : self.loss = val
        else : print name+" is not a parameter, it has te be 'kernel', 'constraint' or 'loss'"
        return

