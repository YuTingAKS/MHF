# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from Transfer_Learning_Methods import FMM_Train_TF

def Main_Function1(DataName, Given, num_user_clusters, num_item_clusters, rating_scales, iteration, split=100000, ratio=0.2,  max_epoches=200, seed=42, tol=1e-2):
    InPut=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName, iteration), allow_pickle=True).tolist()
    if Given==1.0:
        Train_Matrix=np.array(InPut['Train_Matrix'].todense())  
    else:
        Train_Matrix=np.array(InPut['Sparsity{0:.0f}%: Train_Matrix'.format(Given*100,)].todense())
    InPut2=np.load('.\Subspace_Alignment\FMMInitial_{0}_Given{1:.2f}_{2}_{3}_OutPut{4}.npy'.format(\
        DataName, Given, num_user_clusters, num_item_clusters, iteration), allow_pickle=True).tolist()
    PCu0=InPut2['Initial PCu']
    PCv0=InPut2['Initial PCv']
    Pr0=InPut2['Initial Pr']
    FMM=FMM_Train_TF(Train_Matrix, PCu0, PCv0, Pr0, num_user_clusters, num_item_clusters, rating_scales, split, ratio, max_epoches, seed, tol)
    FMM.Main_Function()
    OutPut={}
    OutPut['DataName']=DataName
    OutPut['Given']=Given 
    OutPut['num_user_clusters']=num_user_clusters
    OutPut['num_item_clusters']=num_item_clusters
    OutPut['rating_scales']=rating_scales
    OutPut['split']=split 
    OutPut['ratio']=ratio 
    OutPut['max_epoches']=max_epoches
    OutPut['seed']=seed
    OutPut['tol']=tol  
    OutPut['Initial PCu'], OutPut['Initial PCv'], OutPut['Initial Pr']=FMM.PCu0, FMM.PCv0, FMM.Pr0
    OutPut['Initial PUC'], OutPut['Initial PVC']=FMM.PUC0, FMM.PVC0
    OutPut['AEM: PCu'],OutPut['AEM: PCv'],OutPut['AEM: Pr']=FMM.AEM_PCu,FMM.AEM_PCv,FMM.AEM_Pr
    OutPut['AEM: PUC'],OutPut['AEM: PVC']=FMM.AEM_PUC,FMM.AEM_PVC
    OutPut['AEM: b'],OutPut['AEM: Loss_Array']=FMM.AEM_b,FMM.AEM_Loss
    OutPut['EM: PCu'],OutPut['EM: PCv'],OutPut['EM: Pr']=FMM.EM_PCu,FMM.EM_PCv,FMM.EM_Pr
    OutPut['EM: PUC'],OutPut['EM: PVC']=FMM.EM_PUC,FMM.EM_PVC
    OutPut['EM: Loss_Array']=FMM.EM_Loss
    np.save('.\Subspace_Alignment\FMM_{0}_Given{1:.2f}_{2}_{3}_OutPut{4}.npy'.format(DataName, Given, num_user_clusters, num_item_clusters, iteration), OutPut)
    return

def Main_Function2(DataName, num_user_clusters, num_item_clusters, rating_scales, split=100000, ratio=0.2, max_epoches=200, seed=42, tol=1e-2):
    Train_Matrix=np.array(np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()['After Filtering: Matrix'].todense())
    InPut=np.load('.\Subspace_Alignment\FMMInitial_{0}_{1}_{2}_OutPut.npy'.format(DataName, num_user_clusters, num_item_clusters), allow_pickle=True).tolist()
    PCu0=InPut['Initial PCu']
    PCv0=InPut['Initial PCv']
    Pr0=InPut['Initial Pr']
    FMM=FMM_Train_TF(Train_Matrix, PCu0, PCv0, Pr0, num_user_clusters, num_item_clusters, rating_scales, split, ratio, max_epoches, seed, tol)
    FMM.Main_Function()
    OutPut={}
    OutPut['DataName']=DataName
    OutPut['num_user_clusters']=num_user_clusters
    OutPut['num_item_clusters']=num_item_clusters
    OutPut['rating_scales']=rating_scales
    OutPut['split']=split 
    OutPut['ratio']=ratio 
    OutPut['max_epoches']=max_epoches
    OutPut['seed']=seed
    OutPut['tol']=tol  
    OutPut['Initial PCu'], OutPut['Initial PCv'], OutPut['Initial Pr']=FMM.PCu0, FMM.PCv0, FMM.Pr0
    OutPut['Initial PUC'], OutPut['Initial PVC']=FMM.PUC0, FMM.PVC0
    OutPut['AEM: PCu'],OutPut['AEM: PCv'],OutPut['AEM: Pr']=FMM.AEM_PCu,FMM.AEM_PCv,FMM.AEM_Pr
    OutPut['AEM: PUC'],OutPut['AEM: PVC']=FMM.AEM_PUC,FMM.AEM_PVC
    OutPut['AEM: b'],OutPut['AEM: Loss_Array']=FMM.AEM_b,FMM.AEM_Loss
    OutPut['EM: PCu'],OutPut['EM: PCv'],OutPut['EM: Pr']=FMM.EM_PCu,FMM.EM_PCv,FMM.EM_Pr
    OutPut['EM: PUC'],OutPut['EM: PVC']=FMM.EM_PUC,FMM.EM_PVC
    OutPut['EM: Loss_Array']=FMM.EM_Loss
    np.save('.\Subspace_Alignment\FMM_{0}_{1}_{2}_OutPut.npy'.format(DataName, num_user_clusters, num_item_clusters,), OutPut)
    return  
    


#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    DataName_List=['AmazonBooks',]
    Given_List=[1.0,]
    R_List=[10*x for x in range(4,5)]
    KL_List=[]
    for R in R_List:
        KL_List.append([R,R])
    rating_scales=[x for x in range(1, 6)]
    split=25000
    ratio=0.2
    max_epoches=500
    seed=42
    tol=1e-2 
    for DataName in DataName_List:
        for Given in Given_List:
            for KL in KL_List:
                K,L=KL 
                for iteration in range(1):
                    Main_Function1(DataName, Given, K, L, rating_scales, iteration, split, ratio, max_epoches, seed, tol)
    
    DataName_List=['ML20M', 'Netflix']
    R_List=[10*x for x in range(4,5)]
    KL_List=[]
    for R in R_List:
        KL_List.append([R,R])
    rating_scales=[x for x in range(1, 6)]
    split=25000
    ratio=0.2
    max_epoches=500
    seed=42
    tol=1e-2 
    for DataName in DataName_List:
        for KL in KL_List:
            K,L=KL 
            Main_Function2(DataName, K, L, rating_scales, split, ratio, max_epoches, seed, tol)

