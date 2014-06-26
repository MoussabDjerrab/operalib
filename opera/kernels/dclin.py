from dcpoly import dcpoly

def dclin(X1,X2,B):
    """ decomposable linear kernel B*k_gauss over the scalar gaussian """
    return dcpoly(X1,X2,0,1,B)