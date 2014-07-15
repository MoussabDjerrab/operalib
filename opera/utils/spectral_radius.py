from scipy.sparse import linalg as LAs

def spectralradius(M):
    """
    Give the spectral radius of a matrix M. I.e the maximum of the eigenvalues of M
    """
    l = LAs.eigsh(M, 1,maxiter=1000, return_eigenvectors=False)[0]
    return l