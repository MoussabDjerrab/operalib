# placeholder
import OVKR as OVKRlib

def OVKR(ovkernel="dc",kernel="gauss",c=1,d=1,gamma=1,B="identity",muH=1,normC="L1",muC_1=1,muC_2=1,partitionC=None,partitionC_weight=None):
    return OVKRlib.OVKR(ovkernel,kernel,c,d,gamma,B,muH,normC,muC_1,muC_2,partitionC,partitionC_weight)
def OVKR_gridsearch(X,y,B=5,parameters={}):
    return OVKRlib.grid_search(X,y,B,parameters)
