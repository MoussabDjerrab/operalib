from trpoly import trpoly

def trlin(X1,X2):
    """transformable linear kernel 
        with K(x,x')_ij = k_linear(x(i),x'(j))
    assumes dims(x) = dims(y)
    """
    return trpoly(X1,X2,c=0,d=1)