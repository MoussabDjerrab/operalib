# placeholder
import OVKR as OVKRlib


def OVKR(ovkernel="dc",kernel="gauss",c=1,d=1,gamma=1,B="identity",muH=1,muC=1,normC="L1"):
    return OVKRlib.OVKR(ovkernel,kernel,c,d,gamma,B,muH,muC,normC)
def OVKR_gridsearch(X,y,B=5,parameters={}):
    return OVKRlib.grid_search(X,y,B,parameters)
