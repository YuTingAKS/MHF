import numpy as np 
from copy import deepcopy
import scipy.sparse

def Data_Split(DataName, sparsity_list, seed, fold):
    Input=np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()
    Matrix=Input['After Filtering: Matrix']
    Mask=Input['After Filtering: Matrix_Index']
    num_users,num_items=Matrix.shape 
    userd=np.array(Mask.sum(1).T)[0]
    split=userd/10
    rated_items=list(map(lambda u: Matrix[u].nonzero()[1], range(num_users)))
    rated_users=list(map(lambda u: (np.ones(np.int(userd[u]))*u).astype(np.int), range(num_users)))
    np.random.seed(seed)
    op=list(map(lambda u: np.random.shuffle(rated_items[u]), range(num_users)))
    start1=np.round(split*8).astype(np.int)
    end1=np.round(split*9).astype(np.int)
    start2=np.round(split*9).astype(np.int) 
    end2=np.round(split*10).astype(np.int)
    selected_indices1=np.vstack(list(map(lambda u: np.vstack([rated_users[u][start1[u]:end1[u]], rated_items[u][start1[u]:end1[u]]]).T, \
        range(num_users))))
    selected_indices2=np.vstack(list(map(lambda u: np.vstack([rated_users[u][start2[u]:end2[u]], rated_items[u][start2[u]:end2[u]]]).T, \
        range(num_users))))
    test_num=len(selected_indices1)
    val_num=len(selected_indices2)
    OutPut={}
    OutPut['seed']=seed
    ####  Obtain Test_Mask, Test_Matrix, Train_Mask, Train_Matrix, Test_Rank_Matrix, Test_Rank_Mask  ##############################
    Test_Mask=scipy.sparse.csr_matrix((np.ones(test_num), (selected_indices1[:,0], selected_indices1[:,1])), shape=[num_users, num_items])
    Test_Matrix=Matrix.multiply(Test_Mask)
    Val_Mask=scipy.sparse.csr_matrix((np.ones(val_num), (selected_indices2[:,0], selected_indices2[:,1])), shape=[num_users, num_items])
    Val_Matrix=Matrix.multiply(Val_Mask)
    Train_Mask=Mask-Test_Mask-Val_Mask
    Train_Matrix=Matrix-Test_Matrix-Val_Matrix
    test_ratings=np.array(Matrix[selected_indices1[:,0], selected_indices1[:,1]])[0]
    val_ratings=np.array(Matrix[selected_indices2[:,0], selected_indices2[:,1]])[0]
    ind1=list(filter(lambda x: test_ratings[x]>3, range(test_num)))
    Test_Rank_Mask=scipy.sparse.csr_matrix((np.ones(len(ind1)), (selected_indices1[ind1][:,0], selected_indices1[ind1][:,1])), \
        shape=[num_users, num_items])
    ind2=list(filter(lambda x: val_ratings[x]>3, range(val_num)))
    Val_Rank_Mask=scipy.sparse.csr_matrix((np.ones(len(ind2)), (selected_indices2[ind2][:,0], selected_indices2[ind2][:,1])), \
        shape=[num_users, num_items])
    Test_Rank_Matrix=Matrix.multiply(Test_Rank_Mask)
    Val_Rank_Matrix=Matrix.multiply(Val_Rank_Mask)
    OutPut['Test_Mask'], OutPut['Test_Matrix']=Test_Mask, Test_Matrix 
    OutPut['Test_Rank_Mask'], OutPut['Test_Rank_Matrix']=Test_Rank_Mask, Test_Rank_Matrix
    OutPut['Val_Mask'], OutPut['Val_Matrix']=Val_Mask, Val_Matrix 
    OutPut['Val_Rank_Mask'], OutPut['Val_Rank_Matrix']=Val_Rank_Mask, Val_Rank_Matrix 
    OutPut['Train_Mask'], OutPut['Train_Matrix']=Train_Mask, Train_Matrix
    ####  Extracting x% of data from each target user's training dataset  ######################################################################
    train_userd=np.array(Train_Mask.sum(1).T)[0]
    train_items=list(map(lambda u: Train_Matrix[u].nonzero()[1], range(num_users)))
    train_users=list(map(lambda u: (np.ones(np.int(train_userd[u]))*u).astype(np.int), range(num_users)))
    for sparsity in sparsity_list:
        train_items2=deepcopy(train_items)
        extract_num=np.round(train_userd*sparsity).astype(np.int)
        np.random.seed(seed=42)
        list(map(lambda u: np.random.shuffle(train_items2[u]), range(num_users)))
        extracted_indices=np.vstack(list(map(lambda u: np.vstack([train_users[u][:extract_num[u]], train_items2[u][:extract_num[u]]]).T, \
            range(num_users))))
        Sparsity_Mask=scipy.sparse.csr_matrix((np.ones(len(extracted_indices)), (extracted_indices[:,0], extracted_indices[:,1])), \
            shape=[num_users, num_items])
        Sparsity_Matrix=Train_Matrix.multiply(Sparsity_Mask)
        OutPut['Sparsity{0:.0f}%: Train_Mask'.format(sparsity*100,)]=Sparsity_Mask
        OutPut['Sparsity{0:.0f}%: Train_Matrix'.format(sparsity*100,)]=Sparsity_Matrix
    np.save('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(DataName, fold), OutPut)
    return


def Data_Split_Aux(DataName, seed):
    Input=np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()
    Matrix=Input['After Filtering: Matrix']
    Mask=Input['After Filtering: Matrix_Index']
    num_users,num_items=Matrix.shape 
    userd=np.array(Mask.sum(1).T)[0]
    split=userd/10
    rated_items=list(map(lambda u: Matrix[u].nonzero()[1], range(num_users)))
    rated_users=list(map(lambda u: (np.ones(np.int(userd[u]))*u).astype(np.int), range(num_users)))
    np.random.seed(seed)
    op=list(map(lambda u: np.random.shuffle(rated_items[u]), range(num_users)))
    start1=np.round(split*8).astype(np.int)
    end1=np.round(split*9).astype(np.int)
    selected_indices1=np.vstack(list(map(lambda u: np.vstack([rated_users[u][start1[u]:end1[u]], rated_items[u][start1[u]:end1[u]]]).T, \
        range(num_users))))
    test_num=len(selected_indices1)
    OutPut={}
    OutPut['seed']=seed
    ####  Obtain Test_Mask, Test_Matrix, Train_Mask, Train_Matrix, Test_Rank_Matrix, Test_Rank_Mask  ##############################
    Test_Mask=scipy.sparse.csr_matrix((np.ones(test_num), (selected_indices1[:,0], selected_indices1[:,1])), shape=[num_users, num_items])
    Test_Matrix=Matrix.multiply(Test_Mask)
    Train_Mask=Mask-Test_Mask
    Train_Matrix=Matrix-Test_Matrix
    test_ratings=np.array(Matrix[selected_indices1[:,0], selected_indices1[:,1]])[0]
    ind1=list(filter(lambda x: test_ratings[x]>3, range(test_num)))
    Test_Rank_Mask=scipy.sparse.csr_matrix((np.ones(len(ind1)), (selected_indices1[ind1][:,0], selected_indices1[ind1][:,1])), \
        shape=[num_users, num_items])
    Test_Rank_Matrix=Matrix.multiply(Test_Rank_Mask)
    OutPut['Test_Mask'], OutPut['Test_Matrix']=Test_Mask, Test_Matrix 
    OutPut['Test_Rank_Mask'], OutPut['Test_Rank_Matrix']=Test_Rank_Mask, Test_Rank_Matrix
    OutPut['Train_Mask'], OutPut['Train_Matrix']=Train_Mask, Train_Matrix
    
    np.save('.\Data\{0}_Train_Test_Data.npy'.format(DataName,), OutPut)
    return

if __name__ == '__main__':
    DataName_List=['AmazonBooks']
    seed_list=[42,]
    sparsity_list=[0.25, 0.50, 0.75]
    for DataName in DataName_List:
        for fold,seed in enumerate(seed_list):
            Data_Split(DataName, sparsity_list, seed, fold)
    

    DataName_List=['ML20M', 'Netflix',]
    seed=42
    for DataName in DataName_List:
        Data_Split_Aux(DataName, seed)





