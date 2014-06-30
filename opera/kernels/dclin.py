from dcpoly import dcpoly

def dclin(X1,X2,B):
    """ 
    Decomposable linear kernel  B*k_poly 
    """
    return dcpoly(X1,X2,0,1,B)