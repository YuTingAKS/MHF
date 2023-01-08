import numpy as np 
import scipy.sparse
import pandas as pd
import scipy.sparse.linalg
import scipy.stats
import tensorflow as tf

class PQ_Initialization(object):
    def __init__(self, paths, keys_list, seed, std):
        self.seed=seed
        self.std=std
        InPut=np.load(paths, allow_pickle=True).tolist()
        self.Train_Matrix=InPut[keys_list[0]].todense()
        self.Train_Mask=InPut[keys_list[1]]

    def Initializing(self):
        Matrix=self.Train_Matrix
        indx=Matrix.nonzero()[0]
        indy=Matrix.nonzero()[1]
        ratings=Matrix[indx,indy]
        aver=ratings.mean()
        RM=Matrix-aver
        Pk,a,Qkt=np.linalg.svd(RM, full_matrices=False)
        Qk=Qkt.T
        P=Pk[:,:200]
        Q=Qk[:,:200]
        return P,Q

def Main_Function(DataName, seed, std, Given=' ', iteration=' '):
    if type(Given)!=str and type(iteration)!=str:
        paths='.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName, iteration)
        if Given==1.0:
            keys_list=['Train_Matrix', 'Train_Mask']
        else:
            keys_list=['Sparsity{0:.0f}%: Train_Matrix'.format(Given*100,), 'Sparsity{0:.0f}%: Train_Mask'.format(Given*100,)]
    else:
        paths='.\Data\Filtering_{0}_Results.npy'.format(DataName,)
        keys_list=['After Filtering: Matrix', 'After Filtering: Matrix_Index']
    PQ=PQ_Initialization(paths, keys_list, seed, std)
    P, Q=PQ.Initializing()
    OutPut={}
    OutPut['DataName']=DataName 
    # OutPut['seed'], OutPut['std']=seed, std
    # OutPut['paths'], OutPut['keys_list']=paths, keys_list
    # OutPut['Train_Matrix']=scipy.sparse.csr_matrix(PQ.Train_Matrix)
    # OutPut['Train_Mask']=PQ.Train_Mask ## Already sparse matrix. ##
    OutPut['Initialization P']=P
    OutPut['Initialization Q']=Q
    if type(Given)!=str and type(iteration)!=str:
        np.save('.\Initialization\Initialization_SVD_PQ_{0}_Given{1:.2f}_OutPut{2}.npy'.format(DataName, Given, iteration), OutPut)
    else:
        np.save('.\Initialization\Initialization_SVD_PQ_{0}.npy'.format(DataName,), OutPut)
    return 

#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    DataName_List=['ML20M','Netflix',]
    std=0.1
    seed=42
    for DataName in DataName_List:
        Main_Function(DataName, seed, std)
    
    DataName_List=['AmazonBooks',]
    Given_List=[1.0,]
    std=0.1
    seed=42
    iteration=0
    for DataName in DataName_List:
        for Given in Given_List:
            Main_Function(DataName, seed, std, Given=Given, iteration=iteration)

