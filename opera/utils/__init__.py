from plot import plot_err
from opera.utils.norm import norm
from spectral_radius import spectralradius
from AUC import calc_auc_pr as AUC
from jacobian import jacobian

def vec(M):
    return M.reshape(M.size)