import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from functools import reduce
import os
# start_time=time.clock()

class SAKL(object):
    def __init__(self, DataName_List, Given, R, iteration, seed, std, tol):
        self.R=R 
        self.seed=seed 
        self.std=std
        self.tol=tol
        self.K=len(DataName_List)
        Train_Matrices=[]
        FMM_User=[]
        FMM_Item=[]
        for DataName in DataName_List[:-1]:
            Train_Matrices.append(np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()['After Filtering: Matrix'])
            InPut=np.load('.\Subspace_Alignment\FMM_{0}_{1}_{2}_OutPut.npy'.format(DataName, R, R), allow_pickle=True).tolist()
            Numer1=np.multiply(InPut['EM: PUC'], InPut['EM: PCu'])
            Denom1=Numer1.sum(1)+np.equal(Numer1.sum(1),0.0)
            FMM_User.append(np.true_divide(Numer1.T, Denom1).T)
            Numer2=np.multiply(InPut['EM: PVC'], InPut['EM: PCv'])
            Denom2=Numer2.sum(1)+np.equal(Numer2.sum(1),0.0)
            FMM_Item.append(np.true_divide(Numer2.T, Denom2).T)
        InPut2=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName_List[-1], iteration), allow_pickle=True).tolist()
        if Given==1.0:
            Train_Matrices.append(InPut2['Train_Matrix']) 
        else:
            Train_Matrices.append(InPut2['Sparsity{0:.0f}%: Train_Matrix'.format(Given*100,)])
        InPut3=np.load('.\Subspace_Alignment\FMM_{0}_Given{1:.2f}_{2}_{3}_OutPut{4}.npy'.format(\
            DataName_List[-1], Given, R, R, iteration), allow_pickle=True).tolist()
        Numer1=np.multiply(InPut3['EM: PUC'], InPut3['EM: PCu'])
        Denom1=Numer1.sum(1)+np.equal(Numer1.sum(1),0.0)
        FMM_User.append(np.true_divide(Numer1.T, Denom1).T) 
        Numer2=np.multiply(InPut3['EM: PVC'], InPut3['EM: PCv'])
        Denom2=Numer2.sum(1)+np.equal(Numer2.sum(1),0.0)
        FMM_Item.append(np.true_divide(Numer2.T, Denom2).T)
        self.Train_Matrices=Train_Matrices 
        self.FMM_User=FMM_User 
        self.FMM_Item=FMM_Item
        return

    def Calculating_KL(self, Data):
        NData=[scipy.stats.zscore(x, axis=0) for x in Data]
        Basis=[np.linalg.svd(x.T.dot(x))[0][:,:10] for x in NData] ##basis of each domain##
        TM=[Basis[x].T.dot(Basis[-1]) for x in range(len(Data))] ## The transition matrix for transferring the auxiliary domain's basis into the target domain  ##
        TData=[NData[x].dot(Basis[x]).dot(TM[x]) for x in range(len(Data))] ## Project Data into the target domain's subspace. ##
        Var=[np.var(x,0) for x in TData] ## calculating variance of each column ##
        Std=[np.std(x,0) for x in TData]
        Mean=[np.mean(x,0) for x in TData]
        KL=[np.log(Std[x]/Std[-1])+0.5*(Var[-1]+(Mean[-1]-Mean[x])**2)/Var[x]-0.5 for x in range(len(Data))]
        return KL
    
        
def Main_Function(DataName_List, Given, R, iteration, seed, std, tol):
    SA=SAKL(DataName_List, Given, R, iteration, seed, std, tol)
    KL_User=SA.Calculating_KL(SA.FMM_User)
    KL_Item=SA.Calculating_KL(SA.FMM_Item)
    KL=[np.hstack([KL_User[x], KL_Item[x]]) for x in range(len(DataName_List))]
    DKL=[x.mean() for x in KL]

    Name=reduce(lambda x,y: x+y, [x[0] for x in DataName_List])
    OutPut={}
    OutPut['DataName_List']=DataName_List
    OutPut['Name']=Name
    OutPut['Given']=Given 
    OutPut['R']=R 
    OutPut['seed']=seed 
    OutPut['std']=std 
    OutPut['tol']=tol
    OutPut['KL_User']=KL_User
    OutPut['KL_Item']=KL_Item
    OutPut['Total KL']=KL 
    OutPut['Mean KL']=np.array(DKL)
    np.save('.\Subspace_Alignment\SA_{0}_Given{1:.2f}_R{2}_OutPut{3}.npy'.format(Name, Given, R, iteration), OutPut)
    return 


#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    DataName_List=[['ML20M','Netflix','AmazonBooks',]]
    Given_List=[1.00,]
    R_List=[10*x for x in range(4,5)]
    std=0.1
    seed=42
    tol=1e-2
    for DataName in DataName_List:
        Name=reduce(lambda x,y: x+y, [x[0] for x in DataName])
        for Given in Given_List:
            for R in R_List:
                for iteration in range(1):
                    Main_Function(DataName, Given, R, iteration, seed, std, tol)
                    # path='.\Subspace_Alignment\SA_{0}_Given{1:.2f}_R{2}_OutPut{3}.npy'.format(Name, Given, R, iteration)
                    # if os.path.exists(path):
                    #     continue
                    # else:
                    #     Main_Function(DataName, Given, R, iteration, seed, std, tol)
    
    
    
    