from scipy.sparse import linalg as LAs
import numpy as np

def spectralradius(M):
    """
    Give the spectral radius of a matrix M. I.e the maximum of the eigenvalues of M
    """
    l = 0.
    try:
        l = LAs.eigsh(M, 1,maxiter=1000, return_eigenvectors=False)[0]
    except : 
        l = np.abs(np.linalg.eigvals(M)).max()
    return l