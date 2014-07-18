import numpy as np
import numpy.linalg as LA


def elastic_shooting(K,y,muH=1,muX=1,init=None):
    """ABSTRACT :
        Solve the optimization problem 
            argmin ||y-Kc||^2 + lambda2 * ||h||^2 + lambda1 * ||c||_1
        using subgradient and coordinate descent
        OUTPUT : 
            c       : [n*p] estimation of the solution
            cv      : 1 if convergence, 0 if not
            diff    : (scalar) l1-norm of the difference between the current solution and the previous one 
    """
    (_,Np) = K.shape
    if init is None :
        X = np.linalg.solve(K + muH*np.identity(K.shape[0]), y)
    else :
        X = init.copy()

    m=0
    M=300
    eps = 0.01*LA.norm(y,1)/(Np*1.)

    diff = eps+1

    d = np.zeros(M)

    Z = np.dot(K,(K+muH*np.eye(Np)));

    KTerm = 2*Z
    yTerm = 2*np.dot(K,y)

    while (m < M and diff > eps) :# Stop after M iterations or when the solution is stable
        X_old = X.copy()
        for j in range(Np) : 
            X[j] = 0
            grad = np.dot(KTerm,X) - yTerm #gradient of the Ridge-penalized Residual Sum of Squares
    #         Grad(:,j) = grad;
            if grad[j] > muX : 
                X[j] = (muX-grad[j])/(2*Z[j,j])
            elif grad[j] < -muX :  
                X[j] = (-muX-grad[j])/(2*Z[j,j])
            else : 
                X[j] = 0
        diff = LA.norm(X - X_old,1)
        d[m] = diff
        m=m+1;

    cv = (diff <= eps);
    
    return (X,cv,diff)