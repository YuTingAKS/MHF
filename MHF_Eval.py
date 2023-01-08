import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from functools import reduce
from Evaluation_Test import Evaluation_Function
import os
from multiprocessing import Pool

def Main_Function1(DataName_List, Given, R, T, alpha, lam1, lam2, max_epoches, iteration, topN):
    Name=reduce(lambda x,y: x+y, [DataName[0] for DataName in DataName_List])
    evalpaths='.\MHF\MHF_{0}_Given{1:.2f}_R{2}_T{3}_alpha{4}_lam{5}_lam{6}_Eval_OutPut{7}.npy'.format(\
        Name, Given, R, T, alpha, lam1, lam2, iteration)
    if os.path.exists(evalpaths)==False:
        TargetName=DataName_List[-1]
        Val_Test=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(TargetName,iteration), allow_pickle=True).tolist()
        Test_Matrix=np.array(Val_Test['Test_Matrix'].todense())
        Test_Mask=np.array(Val_Test['Test_Mask'].todense())
        Val_Matrix=np.array(Val_Test['Val_Matrix'].todense())
        Val_Mask=np.array(Val_Test['Val_Mask'].todense())
        Test_Rank_Mask=np.array(Val_Test['Test_Rank_Mask'].todense())
        Test_Rank_Matrix=np.array(Val_Test['Test_Rank_Matrix'].todense())
        Val_Rank_Mask=np.array(Val_Test['Val_Rank_Mask'].todense())
        Val_Rank_Matrix=np.array(Val_Test['Val_Rank_Matrix'].todense())
        if Given==1.0:
            Train_Mask=np.array(Val_Test['Train_Mask'].todense()) 
        else:
            Train_Mask=np.array(Val_Test['Sparsity{0:.0f}%: Train_Mask'.format(Given*100,)].todense())
        num_users=Train_Mask.shape[0]
        num_items=Train_Mask.shape[1]
        Users_List=[np.arange(num_users)]
        UserKeys_List=[' for all users']
        IDCG=np.load('.\Data\{0}_idcg{1}.npy'.format(TargetName, iteration), allow_pickle=True).tolist()
        val_idcg=IDCG['Val: idcg']
        test_idcg=IDCG['Test: idcg']
        OutPut={}
        OutPut['Name']=Name 
        OutPut['Given']=Given 
        OutPut['R'], OutPut['T']=R, T
        OutPut['alpha'], OutPut['lam1'], OutPut['lam2']=alpha, lam1, lam2

        
        paths='.\MHF\MHF_{0}_Given{1:.2f}_R{2}_T{3}_alpha{4}_lam{5}_lam{6}_Epoches{7}_OutPut{8}.npy'.format(\
            Name, Given, R, T, alpha, lam1, lam2, max_epoches, iteration)
        if os.path.exists(paths):
            InPut=np.load(paths, allow_pickle=True).tolist()
            Aver=InPut['Aver'][-1]
            
            U,V,D=InPut['MinRMSE: U'], InPut['MinRMSE: V'], InPut['MinRMSE: D']
            P,Q=InPut['MinRMSE: P'][-1], InPut['MinRMSE: Q'][-1]
            bu,bv=InPut['MinRMSE: bu'][-1], InPut['MinRMSE: bv'][-1]
            A,B=InPut['MinRMSE: A'][-1], InPut['MinRMSE: B'][-1]
            Pred1=np.multiply(P.dot(U), D[-1]).dot(Q.dot(V).T)
            Pred2=A.dot(B.T)+np.tile(bu, [bv.shape[0], 1]).T+np.tile(bv, [bu.shape[0], 1])
            Pred_RMSE=Aver+Pred1+Pred2 

            U,V,D=InPut['MaxNDCG: U'], InPut['MaxNDCG: V'], InPut['MaxNDCG: D']
            P,Q=InPut['MaxNDCG: P'][-1], InPut['MaxNDCG: Q'][-1]
            bu,bv=InPut['MaxNDCG: bu'][-1], InPut['MaxNDCG: bv'][-1]
            A,B=InPut['MaxNDCG: A'][-1], InPut['MaxNDCG: B'][-1]
            Pred1=np.multiply(P.dot(U), D[-1]).dot(Q.dot(V).T)
            Pred2=A.dot(B.T)+np.tile(bu, [bv.shape[0], 1]).T+np.tile(bv, [bu.shape[0], 1])
            Pred_NDCG=Aver+Pred1+Pred2 

            U,V,D=InPut['MaxRecall@20: U'], InPut['MaxRecall@20: V'], InPut['MaxRecall@20: D']
            P,Q=InPut['MaxRecall@20: P'][-1], InPut['MaxRecall@20: Q'][-1]
            bu,bv=InPut['MaxRecall@20: bu'][-1], InPut['MaxRecall@20: bv'][-1]
            A,B=InPut['MaxRecall@20: A'][-1], InPut['MaxRecall@20: B'][-1]
            Pred1=np.multiply(P.dot(U), D[-1]).dot(Q.dot(V).T)
            Pred2=A.dot(B.T)+np.tile(bu, [bv.shape[0], 1]).T+np.tile(bv, [bu.shape[0], 1])
            Pred_Re=Aver+Pred1+Pred2 

            U,V,D=InPut['MaxNDCG@20: U'], InPut['MaxNDCG@20: V'], InPut['MaxNDCG@20: D']
            P,Q=InPut['MaxNDCG@20: P'][-1], InPut['MaxNDCG@20: Q'][-1]
            bu,bv=InPut['MaxNDCG@20: bu'][-1], InPut['MaxNDCG@20: bv'][-1]
            A,B=InPut['MaxNDCG@20: A'][-1], InPut['MaxNDCG@20: B'][-1]
            Pred1=np.multiply(P.dot(U), D[-1]).dot(Q.dot(V).T)
            Pred2=A.dot(B.T)+np.tile(bu, [bv.shape[0], 1]).T+np.tile(bv, [bu.shape[0], 1])
            Pred_NDCGN=Aver+Pred1+Pred2 
            
            Pred_List=[Pred_RMSE, Pred_NDCG, Pred_Re, Pred_NDCGN]
            PredKeys_List=['MinRMSE', 'MaxNDCG', 'MaxRecall@20', 'MaxNDCG@20']
            OutPut=Evaluation_Function(OutPut, Pred_List, PredKeys_List, Train_Mask, Val_Matrix, Val_Rank_Matrix, Val_Mask, Val_Rank_Mask, \
                Test_Matrix, Test_Rank_Matrix, Test_Mask, Test_Rank_Mask, val_idcg, test_idcg, Users_List, UserKeys_List, topN)
            np.save(evalpaths, OutPut)
    return

#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    DataName_List=[['ML20M', 'Netflix', 'AmazonBooks',]]
    Given_List=[1.00,]
    RT_List=[[40, 40,],]
    topN=20
    max_epoches=300
    alpha_list=[1.0,]
    lam1_list=[10,]
    lam2_list=[10,] 
    for DataName in DataName_List:
        for Given in Given_List:
            Name=reduce(lambda x,y: x+y, [z[0] for z in DataName])
            pool = Pool(processes=5)
            for RT in RT_List:
                R,T=RT
                for alpha in alpha_list:
                    for lam1 in lam1_list:
                        for lam2 in lam2_list:
                            for iteration in range(1):
                                pool.apply_async(Main_Function1, \
                                    args=(DataName, Given, R, T, alpha, lam1, lam2, max_epoches, iteration, topN))
            pool.close()
            pool.join()

    