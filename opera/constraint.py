import numpy as np
import numpy.linalg as LA
from opera.utils import query_yes_no

class constraint():
    """
    """

    def __init__(self,reg="lasso",lambda1=1,lambda2=1,partition=[],weight_partition=[]):
        self.reg = reg
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.partition = partition
        self.weight_partition = weight_partition
        self.gradients_ = []
        self.functions_ = []

        if reg.upper() == 'L1' or reg.lower() == 'lasso' :
            self.add_function(lambda x : lambda1*LA.norm(x,1),False)
        elif reg.upper() == 'L2' :
            self.add_function(lambda x : lambda2*LA.norm(x,2),False)
        elif reg.lower() == 'elasticnet'or reg.lower() == 'elastic net' :
            self.add_function(lambda x : lambda1*LA.norm(x,1) + lambda2*LA.norm(x,2) ,False)
        elif reg.lower() == 'mixed' or reg.lower() == 'grouplasso'or reg.lower() == 'group lasso' :
            self.add_function(lambda x : fun_mixed(x,lambda1,partition,weight_partition) ,False)
        elif reg.lower() == 'sparsemixed' or reg.lower() == 'sparsegrouplasso'or reg.lower() == 'sparse group lasso' or reg.lower() == 'sparse mixed' :
            self.add_function(lambda x : lambda1*LA.norm(x,1) + fun_mixed(x,lambda2,partition,weight_partition) ,False)

    def __repr__(self):
        "Constraint : < "+self.reg+">"
    def __str__(self):
        self.reg
    def copy(self):
        c = constraint(self.reg,self.lambda1,self.lambda2,self.partition,self.weight_partition)
        c.gradients_ = self.gradients_
        c.gradients_ = self.functions_
        return c

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

    def add_function(self,f,is_smooth,grad=None):
        if is_smooth and grad is None :
            if query_yes_no("Warning : no gradient specified, make it zero ?") :
                grad = lambda x : 0
            else : return
        self.functions_.append((f,is_smooth))
        if is_smooth :
            self.gradients_.append(grad)
        return



    def prox_operator(self):
        """
        use proximal methode on a x
        self.reg :
            L1 or lasso : lambda1 ||C||_1
            L2 : lambda2 ||C||_2
            elastic-net : lambda1 ||C||_1 + lambda2 ||C||2^2
            group lasso : lambda1 sum ||C||_2 with partition a set of index set weight_part the weight of each group (index set).
            sparse groupe lasso : lambda1||C||_1 + lambda2 sum ||C||_2 with partition a set of index set weight_part the weight of each group (index set).
        """
        if self.reg.upper() == 'L1' or self.reg.lower() == 'lasso' :
            return lambda x , l : prox_lasso(x,self.lambda1*l)
        elif self.reg.upper() == 'L2' :
            return lambda x , l : prox_l2(x,self.lambda2*l)
        elif self.reg.lower() == 'elasticnet'or self.reg.lower() == 'elastic net' :
            return lambda x , l : prox_elasticnet(x,self.lambda1*l,self.lambda2*l)
        elif self.reg.lower() == 'mixed' or self.reg.lower() == 'grouplasso'or self.reg.lower() == 'group lasso' :
            return lambda x , l : prox_grouplasso(x,self.partition,self.weight_partition*l)
        elif self.reg.lower() == 'sparsemixed' or self.reg.lower() == 'sparsegrouplasso'or self.reg.lower() == 'sparse group lasso' or self.reg.lower() == 'sparse mixed' :
            return lambda x , l : prox_sparsegroupelasso(x,self.lambda1*l,self.partition,self.weight_partition*l)
        else : print("Warning in constraint : reg "+self.reg+" unknow")
        return lambda x , l : x


def fun_mixed(x,lambda2,partition,weight_partition):
    if partition is None :
        return lambda2*LA.norm(x,2)
    elif weight_partition is None :
        weights = np.ones(len(partition))
    sol = 0
    #for each parition we extract the group ug
    n_group = 0 # the number of group, usefull to know what group it is for weight
    for group in partition :
        u = np.zeros(len(group))
        j = 0
        for i in group :
            u[j] = x[i]
            j = j+1
        norm2_u = LA.norm(u,2)
        sol += weights[n_group] * norm2_u
        n_group +=1
    return sol

# prox_op definition
def prox_lasso(x,lambda1):
    ''' l1-norm regularization
    [Prox_[lambda||.||_1] (u) ]j = (1-lambda/|uj|) uj = sgn(uj)(|uj|-lambda)_+
    '''
    tmp2 = np.abs(x)-lambda1
    # test = (tmp2>=0) but it is not working, i don't know why
    test = tmp2.copy()
    test[tmp2>=0] = 1
    test[tmp2<0] = 0
    Sol = test*tmp2*np.sign(x)
    return Sol

def prox_l2(x,lambda2):
    '''
    Prox_[lambda/2||.||_2^2](u) = u /(1+lambda)
    '''
    return (1/(1+2*lambda2))*x

def prox_elasticnet(x,lambda1,lambda2):
    ''' l1+l2^2-regularization
    Prox_lambda(||.||_1+gamma/2||.||_2^2) = Prox_[gammalambda/2||.||_2^2] o Prox_[lambda||.||_1] = 1/(1+lambdagamma) Prox_[lambda||.||_1]
    '''
    return prox_l2( prox_lasso(x,lambda1) , lambda2)

def prox_grouplasso(x,partition,weights):
    ''' l1/l2-norm regularization
    [Prox_lambda(u)]_g = ( 1 - lambda / ||u_g||_2 )_+ * u_g where g in partition
    '''
    if partition is None :
        return prox_l2(x,1)
    elif weights is None :
        weights = np.ones(len(partition))

    sol = np.zeros(len(x))
    #for each parition we extract the group ug
    n_group = 0 # the number of group, usefull to know what group it is for weight
    for group in partition :
        u = np.zeros(len(group))
        j = 0
        for i in group :
            u[j] = x[i]
            j = j+1
        norm2_u = LA.norm(u,2)
        if norm2_u != 0 :
            j = 0
            for i in group :
                c = (1-weights[n_group]/norm2_u)
                if c < 0 : sol[i] = 0
                else :  sol[i] = (1-weights[n_group]/norm2_u)*u[j]
                j = j+1
        n_group+=1
    return sol

def prox_sparsegroupelasso(x,lambda1,partition,weights):
    '''Combined l1+l1/l2-norm l1+l2^2-regularization
    '''
    return prox_grouplasso( prox_lasso(x,lambda1) ,partition,weights)
