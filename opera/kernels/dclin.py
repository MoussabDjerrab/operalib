from dcpoly import __init__ as dcpoly

def __init__(X1,X2,B):
    """ decomposable linear kernel B*k_gauss over the scalar gaussian """
    return dcpoly(X1,X2,0,1,B)