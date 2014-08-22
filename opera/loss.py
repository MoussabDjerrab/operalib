import numpy as np
from opera.utils import query_yes_no

class loss():
    """
    """

    def __init__(self):
        self.gradients_ = []
        self.functions_ = []

    def __repr__(self):
        "Loss : < "+self.functions.size+" functions  >"
    def __str__(self):
        self.reg
    def copy(self):
        l = loss()
        l.gradients_ = self.gradients_
        l.functions_ = self.functions_

    def gradients(self):
        """
        Give the gradient of the functions. If is not smooth then it will be 0
        """
        return np.array(self.gradients_)

    def functions(self):
        """
        Give the the functions. each elements are (function,is_smooth)
        """
        return np.array(self.functions_)

    def add_function(self,f,grad=None):
        if grad is None :
            if query_yes_no("Warning : no gradient specified, make it zero ?") :
                grad = lambda x : 0
            else : return
        self.functions_.append(f)
        self.gradients_.append(grad)
        return
