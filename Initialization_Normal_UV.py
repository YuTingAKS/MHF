import numpy as np 
import scipy.sparse
import pandas as pd
import scipy.sparse.linalg
import scipy.stats
# from multiprocessing import Pool
# import matplotlib.pyplot as plt 
# from matplotlib.pylab import *
# start_time=time.clock()

class UV_Initializatoin(object):
    def __init__(self, R_List, seed, std):
        self.R_List=R_List 
        self.seed=seed 
        self.std=std 
    
    def Intializing(self):
        U0={}
        V0={}
        for R in self.R_List:
            np.random.seed(self.seed)
            U0['R={0}'.format(R,)]=np.random.normal(0, self.std, R*R).reshape(-1, R)
            V0['R={0}'.format(R,)]=np.random.normal(0, self.std, R*R).reshape(-1, R)
        self.U_List=U0 
        self.V_List=V0
        return U0,V0

        
def Main_Function(R_List, seed, std):
    UV=UV_Initializatoin(R_List, seed, std)
    U_List,V_List=UV.Intializing()
    OutPut={}
    OutPut['R_List']=R_List 
    OutPut['seed'], OutPut['std']=seed, std
    for R in R_List:
        OutPut['Initialization U: R={0}'.format(R,)]=U_List['R={0}'.format(R,)]
        OutPut['Initialization V: R={0}'.format(R,)]=V_List['R={0}'.format(R,)]
    np.save('.\Initialization\Initialization_Normal_UV.npy', OutPut)
    return 


#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    R_List=[10*x for x in range(1,21)]
    std=0.1
    seed=42
    Main_Function(R_List, seed, std)
    
    
