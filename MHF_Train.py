import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from functools import reduce
from Transfer_Learning_Methods import MHF_TFCP 
import os

def Main_Function1(DataName_List, Given, R, T, alpha, lam1, lam2, iteration, max_epoches=500, seed=42, std=0.1, tol=1e-2):
    Name=reduce(lambda x,y: x+y, [x[0] for x in DataName_List])
    Train_Matrices=[]
    Train_Masks=[]
    for DataName in DataName_List[:-1]:
        InPut1=np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()
        Train_Matrices.append(np.array(InPut1['After Filtering: Matrix'].todense()))
        Train_Masks.append(np.array(InPut1['After Filtering: Matrix_Index'].todense()))
    InPut2=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName_List[-1], iteration), allow_pickle=True).tolist()
    Val_Matrix=np.array(InPut2['Val_Matrix'].todense())
    Val_Mask=np.array(InPut2['Val_Mask'].todense())
    Val_Rank_Mask=np.array(InPut2['Val_Rank_Mask'].todense())
    Val_Rank_Matrix=np.array(InPut2['Val_Rank_Matrix'].todense())
    if Given==1.0:
        Train_Matrices.append(np.array(InPut2['Train_Matrix'].todense()))
        Train_Masks.append(np.array(InPut2['Train_Mask'].todense()))
    else:
        Train_Matrices.append(np.array(InPut2['Sparsity{0:.0f}%: Train_Matrix'.format(Given*100,)].todense()))
        Train_Masks.append(np.array(InPut2['Sparsity{0:.0f}%: Train_Mask'.format(Given*100,)].todense()))
    InPut3=np.load('.\Subspace_Alignment\SA_{0}_Given{1:.2f}_R{2}_OutPut{3}.npy'.format(Name, Given, 40, iteration), allow_pickle=True).tolist()
    DKL=InPut3['Mean KL']
    weights=np.exp(-alpha*DKL)
    D0=np.ones((len(DataName_List), R))
    InPut4=np.load('.\Initialization\Initialization_Normal_UV.npy', allow_pickle=True).tolist()
    U0=InPut4['Initialization U: R={0}'.format(R,)]
    V0=InPut4['Initialization V: R={0}'.format(R,)]
    P0=[]
    Q0=[]
    for DataName in DataName_List[:-1]:
        InPut5=np.load('.\Initialization\Initialization_SVD_PQ_{0}.npy'.format(DataName,), allow_pickle=True).tolist()
        P0.append(InPut5['Initialization P'][:,:R])
        Q0.append(InPut5['Initialization Q'][:,:R])
    InPut6=np.load('.\Initialization\Initialization_SVD_PQ_{0}_Given{1:.2f}_OutPut{2}.npy'.format(DataName_List[-1], Given, iteration), allow_pickle=True).tolist()
    P0.append(InPut6['Initialization P'][:,:R])
    Q0.append(InPut6['Initialization Q'][:,:R])
    MHF=MHF_TFCP(Train_Matrices, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Matrix, Val_Rank_Mask, \
        U0, V0, D0, P0, Q0, weights, T, lam1, lam2, max_epoches, seed, tol)
    MHF.Transfer_Learning()
    OutPut={}
    OutPut['Name']=Name 
    OutPut['Given']=Given 
    OutPut['R'], OutPut['T']=R,T
    OutPut['alpha'], OutPut['lam1'], OutPut['lam2']=alpha, lam1, lam2
    OutPut['seed']=seed 
    OutPut['std']=std
    OutPut['tol']=tol
    OutPut['max_epoches']=max_epoches
    OutPut['Weights']=weights
    OutPut['Loss_Array']=MHF.Loss_Array
    OutPut['Aver']=MHF.Aver
    OutPut['P0'], OutPut['Q0']=MHF.P0, MHF.Q0 
    OutPut['U0'], OutPut['V0'], OutPut['D0']=MHF.U0, MHF.V0, MHF.D0 
    OutPut['bu0'], OutPut['bv0']=MHF.bu0, MHF.bv0
    OutPut['A0'], OutPut['B0']=MHF.A0, MHF.B0
    OutPut['P'], OutPut['Q']=MHF.P, MHF.Q 
    OutPut['U'], OutPut['D'], OutPut['V']=MHF.U, MHF.D, MHF.V 
    OutPut['bu'], OutPut['bv']=MHF.bu, MHF.bv 
    OutPut['A'], OutPut['B']=MHF.A, MHF.B
    OutPut['MinRMSE: IterTime']=MHF.MinRMSE_IterTime 
    OutPut['MinRMSE: P'], OutPut['MinRMSE: Q']=MHF.MinRMSE_P, MHF.MinRMSE_Q
    OutPut['MinRMSE: U'], OutPut['MinRMSE: D'], OutPut['MinRMSE: V']=MHF.MinRMSE_U, MHF.MinRMSE_D, MHF.MinRMSE_V
    OutPut['MinRMSE: bu'], OutPut['MinRMSE: bv']=MHF.MinRMSE_bu, MHF.MinRMSE_bv
    OutPut['MinRMSE: A'], OutPut['MinRMSE: B']=MHF.MinRMSE_A, MHF.MinRMSE_B
    OutPut['MaxNDCG: IterTime']=MHF.MaxNDCG_IterTime 
    OutPut['MaxNDCG: P'], OutPut['MaxNDCG: Q']=MHF.MaxNDCG_P, MHF.MaxNDCG_Q
    OutPut['MaxNDCG: U'], OutPut['MaxNDCG: D'], OutPut['MaxNDCG: V']=MHF.MaxNDCG_U, MHF.MaxNDCG_D, MHF.MaxNDCG_V
    OutPut['MaxNDCG: bu'], OutPut['MaxNDCG: bv']=MHF.MaxNDCG_bu, MHF.MaxNDCG_bv
    OutPut['MaxNDCG: A'], OutPut['MaxNDCG: B']=MHF.MaxNDCG_A, MHF.MaxNDCG_B 
    OutPut['MaxRecall@20: IterTime']=MHF.MaxRe20_IterTime 
    OutPut['MaxRecall@20: P'], OutPut['MaxRecall@20: Q']=MHF.MaxRe20_P, MHF.MaxRe20_Q
    OutPut['MaxRecall@20: U'], OutPut['MaxRecall@20: D'], OutPut['MaxRecall@20: V']=MHF.MaxRe20_U, MHF.MaxRe20_D, MHF.MaxRe20_V
    OutPut['MaxRecall@20: bu'], OutPut['MaxRecall@20: bv']=MHF.MaxRe20_bu, MHF.MaxRe20_bv
    OutPut['MaxRecall@20: A'], OutPut['MaxRecall@20: B']=MHF.MaxRe20_A, MHF.MaxRe20_B
    OutPut['MaxNDCG@20: IterTime']=MHF.MaxNDCG20_IterTime 
    OutPut['MaxNDCG@20: P'], OutPut['MaxNDCG@20: Q']=MHF.MaxNDCG20_P, MHF.MaxNDCG20_Q
    OutPut['MaxNDCG@20: U'], OutPut['MaxNDCG@20: D'], OutPut['MaxNDCG@20: V']=MHF.MaxNDCG20_U, MHF.MaxNDCG20_D, MHF.MaxNDCG20_V
    OutPut['MaxNDCG@20: bu'], OutPut['MaxNDCG@20: bv']=MHF.MaxNDCG20_bu, MHF.MaxNDCG20_bv 
    OutPut['MaxNDCG@20: A'], OutPut['MaxNDCG@20: B']=MHF.MaxNDCG20_A, MHF.MaxNDCG20_B
    
    np.save('.\MHF\MHF_{0}_Given{1:.2f}_R{2}_T{3}_alpha{4}_lam{5}_lam{6}_Epoches{7}_OutPut{8}.npy'.format(\
        Name, Given, R, T, alpha, lam1, lam2, max_epoches, iteration,), OutPut)
    return

#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    DataName_List=[['ML20M', 'Netflix', 'AmazonBooks',],]
    Given_List=[1.00,]
    RT_List=[[40, 40],]
    tol=1e-2
    seed=42
    std=0.1
    max_epoches=300

    alpha_list=[1.00,]
    lam1_list=[10,] 
    lam2_list=[10,] 
    for DataName in DataName_List:
        Name=reduce(lambda x,y: x+y, [z[0] for z in DataName])
        for Given in Given_List:
            for RT in RT_List:
                R,T=RT
                for alpha in alpha_list:
                    for lam1 in lam1_list:
                        for lam2 in lam2_list:
                            for iteration in range(1):
                                Flag=0
                                path1='.\MHF\MHF_{0}_Given{1:.2f}_R{2}_T{3}_alpha{4}_lam{5}_lam{6}_Epoches{7}_OutPut{8}.npy'.format(\
                                    Name, Given, R, T, alpha, lam1, lam2,  max_epoches, iteration)
                                if os.path.exists(path1):
                                    Flag=1
                                if Flag==0:
                                    Main_Function1(DataName, Given, R, T, alpha, lam1, lam2, iteration, max_epoches, seed, std, tol)
    

    
    