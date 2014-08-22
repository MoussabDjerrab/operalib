'''
.. module:: OVKR
   :platform: Unix, Windows
   :synopsis: module to performs an OVKR

.. moduleauthor:: Tristan Tchilinguirian <tristan.tchilinguirian@ensiie.fr>

Link prediction is addressed as an output kernel learning task through semi-supervised Output Kernel Regression.
Working in the framework of RKHS theory with vector-valued functions, we establish a new representer theorem devoted to semi-supervised least square regression. We then apply it to get a new model (POKR: Penalized Output Kernel Regression) and show its relevance using numerical experiments on artificial networks and two real applications using a very low percentage of labeled data in a transductive setting.

    * Celine Bouard, Florence d'Alche-Buc and Marie Szafranski (2011) Semi-Supervized Penalized Output Kernel Regression for Link Prediction. In ICML 2011.
'''

from .OPERAObject import OPERAObject
import numpy as np
from opera import proximal


class OVKR(OPERAObject):
    """
    Performs OVK regression over parameter ranges, cross-validation, etc.

    :param kernel:
    :type opera.kernels.Kernel
    :param constraint:
    :type opera.constraint
    :param loss:
    :type opera.loss
    """

    def __init__(self, kernel, constraint, loss = None):
        super(OVKR,self).__init__(kernel, constraint, loss)

    def __repr__(self):
        if self.kernel.K is None : fitted = "no "
        else : fitted = "yes"
        return "OVKR model : < fitted:"+fitted+" >"
    def __str__(self):
        out = "OVKR model : \n"
        #parameters print
        out += str(self.kernel)+"\n"
        out += str(self.constraint)+"\n"
        out += str(self.loss)+"\n"
        return out
    def copy(self):
        return OVKR(self.kernel.copy(),self.constraint.copy(),self.loss.copy())


    def fit(self, X, y):
        """Method to fit a model

        :param: array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        :type: ndarray
        :param: array, with shape = [N,p], where N is the number of samples.
        :type: ndarray
        """
        OPERAObject.fit(self, X, y)
        # compute the kernel
        self.kernel.compute_matrix(X,X,y)
        # learning C
        Yvec = np.reshape(y, (len(y[0,:])*len(y[:,0])))
        Cvec = proximal.proximalLinear(self.kernel.matrix(), Yvec, self.constraint, eps=1.e-3)
        self.C = np.reshape(Cvec,y.T.shape).T


    def predict(self,X):
        """Method to predict theclust of a data

        :param: array-like, with shape = [N, D], where N is the number of samples and D is the number of features.
        :type: ndarray
        :returns: array, with shape = [N,p], where N is the number of samples.
        :rtype: ndarray
        """
        Ktest = self.kernel.compute_matrix(X,self.datas,self.labels,False)
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


