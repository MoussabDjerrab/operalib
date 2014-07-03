import numpy as np
import numpy.linalg as LA

def proximal(gradient,norm='L1',mu1=1,mu2=1,partition=None,weight_partition=None):
    """
    use proximal methode on a gradient
    norm : 
        L1 or lasso : mu1 ||C||_1
        L2 : mu2 ||C||_2
        elastic-net : mu1 ||C||_1 + mu2 ||C||2^2 
        group lasso : mu1 sum ||C||_2 with partition a set of index set weight_part the weight of each group (index set).
        sparse groupe lasso : mu1||C||_1 + mu2 sum ||C||_2 with partition a set of index set weight_part the weight of each group (index set).
    """
    if norm.upper() == 'L1' or norm.lower() == 'lasso' : 
        return prox_lasso(gradient,mu1)
    
    elif norm.upper == 'L2' : 
        return prox_l2(gradient,mu2)
    
    elif norm.lower() == 'elasticnet'or norm.lower() == 'elastic net' : 
        return prox_electicnet(gradient,mu1,mu2)
    
    elif norm.lower() == 'mixed' or norm.lower() == 'grouplasso'or norm.lower() == 'group lasso' : 
        return prox_grouplasso(gradient,partition,weight_partition)
    
    elif norm.lower() == 'sparsemixed' or norm.lower() == 'sparsegrouplasso'or norm.lower() == 'sparse group lasso' or norm.lower() == 'sparse mixed' :
        return prox_sparsegroupelasso(gradient,mu1,partition,weight_partition)
  
    return



def prox_lasso(grad,mu):
    ''' l1-norm regularization
    [Prox_[mu||.||_1] (u) ]j = (1-mu/|uj|) uj = sgn(uj)(|uj|-mu)_+
    '''
    tmp2 = np.abs(grad)-mu
    # test = (tmp2>=0) but it is not working, i don't know why
    test = tmp2.copy()
    test[tmp2>=0] = 1
    test[tmp2<0] = 0
    Sol = test*tmp2*np.sign(grad)
    return Sol

def prox_l2(grad,mu):
    '''
    Prox_[mu/2||.||_2^2](u) = u /(1+mu) 
    '''
    return (1/(1+2*mu))*grad

def prox_electicnet(grad,mu1,mu2):
    ''' l1+l2^2-regularization
    Prox_mu(||.||_1+gamma/2||.||_2^2) = Prox_[gammamu/2||.||_2^2] o Prox_[mu||.||_1] = 1/(1+mugamma) Prox_[mu||.||_1]
    '''
    return prox_l2( prox_lasso(grad,mu1) , mu2)

def prox_grouplasso(grad,partition,weights):
    ''' l1/l2-norm regularization
    [Prox_mu(u)]_g = ( 1 - mu / ||u_g||_2 )_+ * u_g where g in partition
    '''
    if partition is None : 
        return prox_lasso(grad,1)
    elif weights is None :
        weights = np.ones(len(partition))
    
    sol = np.zeros(len(grad))
    #for each parition we extract the group ug
    n_group = 0 # the number of group, usefull to know what group it is for weight
    for group in partition : 
        u = np.zeros(len(group))
        j = 0
        for i in group : 
            u[j] = grad[i]
            j = j+1
        norm2_u = LA.norm(u,2)
        if norm2_u != 0 : 
            j = 0
            for i in group : 
                c = (1-weights[n_group]/norm2_u)
                if c < 0 : sol[i] = 0
                else :  sol[i] = (1-weights[n_group]/norm2_u)*u[j]
                j = j+1
    return sol

def prox_sparsegroupelasso(grad,mu,partition,weights):
    '''Combined l1+l1/l2-norm l1+l2^2-regularization
    '''
    return prox_grouplasso( prox_lasso(grad,mu) ,partition,weights)
