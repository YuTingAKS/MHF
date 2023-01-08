import numpy as np
import scipy.sparse
import os
from Transfer_Learning_Methods import CodeBook_Construction

def Main_Function1(DataName, Given, num_user_clusters, num_item_clusters, rating_scales, iteration, max_epoches=200, seed=42, tol=1e-2):
    InPut=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName, iteration), allow_pickle=True).tolist()
    if Given==1.0:
        Train_Matrix=np.array(InPut['Train_Matrix'].todense())
    else:
        Train_Matrix=np.array(InPut['Sparsity{0:.0f}%: Train_Matrix'.format(Given*100,)].todense())
    CBC=CodeBook_Construction(Train_Matrix, num_user_clusters, num_item_clusters, rating_scales, max_epoches, seed, tol)
    CodeBook=CBC.Generating_CodeBook()
    OutPut={}
    OutPut['DataName']=DataName 
    OutPut['num_user_clusters']=num_user_clusters 
    OutPut['num_item_clusters']=num_item_clusters
    OutPut['rating_scales']=rating_scales 
    OutPut['max_epoches']=max_epoches 
    OutPut['seed']=seed 
    OutPut['tol']=tol 
    OutPut['ONMTF: U'], OutPut['ONMTF: S'], OutPut['ONMTF: V']=CBC.U, CBC.S, CBC.V
    OutPut['ONMTF: Loss_Array']=CBC.Error_Array 
    OutPut['User_Clusters'], OutPut['Item_Clusters']=scipy.sparse.csr_matrix(CBC.U_C), scipy.sparse.csr_matrix(CBC.V_C) 
    OutPut['CodeBook']=CodeBook
    U_C,V_C=scipy.sparse.csr_matrix(CBC.U_C), scipy.sparse.csr_matrix(CBC.V_C)
    PCu=np.array(U_C.sum(0)/U_C.sum())[0]
    PCv=np.array(V_C.sum(0)/V_C.sum())[0]
    userind,itemind=Train_Matrix.nonzero()
    ratings=Train_Matrix[userind, itemind]
    indices=[list(filter(lambda x: ratings[x]==r, range(len(ratings)))) for r in rating_scales]
    RM_List=[scipy.sparse.csr_matrix((np.ones(len(index)), (userind[index], itemind[index])), shape=Train_Matrix.shape) for index in indices]
    Pr_Numer=[np.array((U_C.T.dot(RM).dot(V_C)).todense()).reshape(1, num_user_clusters, num_item_clusters) for RM in RM_List]
    Pr_Denom0=np.concatenate(Pr_Numer, axis=0).sum(0)
    Pr_Denom=Pr_Denom0+np.equal(Pr_Denom0, 0.0)
    Pr=np.concatenate([np.true_divide(Numer, Pr_Denom) for Numer in Pr_Numer], axis=0)
    OutPut['Initial PCu']=PCu 
    OutPut['Initial PCv']=PCv 
    OutPut['Initial Pr']=Pr
    np.save('.\Subspace_Alignment\FMMInitial_{0}_Given{1:.2f}_{2}_{3}_OutPut{4}.npy'.format(DataName, Given, num_user_clusters, num_item_clusters, iteration), OutPut)
    return 

def Main_Function2(DataName, num_user_clusters, num_item_clusters, rating_scales, max_epoches=200, seed=42, tol=1e-2):
    Matrix=np.array(np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()['After Filtering: Matrix'].todense())
    CBC=CodeBook_Construction(Matrix, num_user_clusters, num_item_clusters, rating_scales, max_epoches, seed, tol)
    CodeBook=CBC.Generating_CodeBook()
    OutPut={}
    OutPut['DataName']=DataName 
    OutPut['num_user_clusters']=num_user_clusters 
    OutPut['num_item_clusters']=num_item_clusters
    OutPut['rating_scales']=rating_scales 
    OutPut['max_epoches']=max_epoches 
    OutPut['seed']=seed 
    OutPut['tol']=tol 
    OutPut['ONMTF: U'], OutPut['ONMTF: S'], OutPut['ONMTF: V']=CBC.U, CBC.S, CBC.V
    OutPut['ONMTF: Loss_Array']=CBC.Error_Array 
    OutPut['User_Clusters'], OutPut['Item_Clusters']=scipy.sparse.csr_matrix(CBC.U_C), scipy.sparse.csr_matrix(CBC.V_C) 
    OutPut['CodeBook']=CodeBook
    U_C,V_C=scipy.sparse.csr_matrix(CBC.U_C), scipy.sparse.csr_matrix(CBC.V_C)
    PCu=np.array(U_C.sum(0)/U_C.sum())[0]
    PCv=np.array(V_C.sum(0)/V_C.sum())[0]
    userind,itemind=Matrix.nonzero()
    ratings=Matrix[userind, itemind]
    indices=[list(filter(lambda x: ratings[x]==r, range(len(ratings)))) for r in rating_scales]
    RM_List=[scipy.sparse.csr_matrix((np.ones(len(index)), (userind[index], itemind[index])), shape=Matrix.shape) for index in indices]
    Pr_Numer=[np.array((U_C.T.dot(RM).dot(V_C)).todense()).reshape(1, num_user_clusters, num_item_clusters) for RM in RM_List]
    Pr_Denom0=np.concatenate(Pr_Numer, axis=0).sum(0)
    Pr_Denom=Pr_Denom0+np.equal(Pr_Denom0, 0.0)
    Pr=np.concatenate([np.true_divide(Numer, Pr_Denom) for Numer in Pr_Numer], axis=0)
    OutPut['Initial PCu']=PCu 
    OutPut['Initial PCv']=PCv 
    OutPut['Initial Pr']=Pr
    np.save('.\Subspace_Alignment\FMMInitial_{0}_{1}_{2}_OutPut.npy'.format(DataName, num_user_clusters, num_item_clusters), OutPut)
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
    max_epoches=500
    seed=42
    tol=1e-2
    for DataName in DataName_List:
        for Given in Given_List:
            for KL in KL_List:
                K,L=KL
                for iteration in range(1):
                    Main_Function1(DataName, Given, K, L, rating_scales, iteration, max_epoches, seed, tol)

    DataName_List=['ML20M', 'Netflix']
    R_List=[10*x for x in range(4,5)]
    KL_List=[]
    for R in R_List:
        KL_List.append([R,R])
    rating_scales=[x for x in range(1, 6)]
    max_epoches=500
    seed=42
    tol=1e-2
    for DataName in DataName_List:
        for KL in KL_List:
            K,L=KL
            Main_Function2(DataName, K, L, rating_scales, max_epoches, seed, tol)
    
