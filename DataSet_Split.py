import numpy as np 
import scipy.sparse 
import xlwt 
# import matplotlib.pyplot as plt 
# from matplotlib.pylab import *
def filteringusers_1ratingsOr1items(Array, Matrix, Matrix_Index, table, row,):
    ##filter out users who only rate one item or rate all items who one rating.##
    num_users,num_items=Matrix.shape 
    Means=np.array(Matrix.sum(1).T)[0]/np.array(Matrix_Index.sum(1).T)[0]
    New_Array=np.zeros((len(Array),3))
    New_Array[:,0]=Array[:,0]-Array[:,0].min()
    New_Array[:,1]=Array[:,1]
    New_Array[:,2]=Array[:,2]-Means[(Array[:,0]-Array[:,0].min()).astype(int)]
    New_Matrix=scipy.sparse.csr_matrix((New_Array[:,2], (New_Array[:,0].astype(int), New_Array[:,1].astype(int))), \
        shape=(num_users,num_items))
    users=np.array(New_Matrix.sum(1).T)[0].nonzero()[0] ## users whose adopt at least two unique ratings to rate items. ##
    num_users0=num_users-len(users) ##the number of users who are filtered out ##
    Matrix1=Matrix[users]
    Matrix_Index1=Matrix_Index[users]
    items=np.array(Matrix_Index1.sum(0))[0].nonzero()[0]
    Matrix2=Matrix1[:,items]
    Matrix_Index2=Matrix_Index1[:,items]
    num_users,num_items=Matrix_Index2.shape
    num_ratings=Matrix_Index2.sum()
    density=num_ratings/(num_users*num_items)
    table.write(row,0,'Filtering out users who have only rated one items or rated all items only with one ratings')
    row+=1
    table.write(row,0,'After Filtering')
    table.write(row,1,'number of removed users')
    table.write(row,2,num_users0)
    row+=1
    table.write(row,1,'num_users')
    table.write(row,2,num_users)
    row+=1
    table.write(row,1,'num_items')
    table.write(row,2,num_items)
    row+=1
    table.write(row,1,'num_ratings')
    table.write(row,2,num_ratings)
    row+=1
    table.write(row,1,'average_ratings')
    table.write(row,2,Matrix2.sum()/num_ratings)
    row+=1
    table.write(row,1,'density')
    table.write(row,2,density)
    row+=1
    return Matrix2, Matrix_Index2, table, row,

def filtering_user_items_percentile(Matrix, Matrix_Index, table, row, split_user, split_item):
    num_users,num_items=Matrix.shape
    itemd=np.array(Matrix_Index.sum(0))[0].astype(np.int)
    itemd_keys=np.unique(itemd)
    table.write(row,0,'calculating among the keys of item degrees')
    table.write(row,1,'percentile')
    table.write(row,2,'item degree')
    table.write(row,3,'item degree range')
    table.write(row,4,'percentage of items')
    table.write(row,5,'cumulative percentage')
    table.write(row,6,'number of items whose item degrees are larger than the given item degree')
    itemd_key_per=[[]]*21
    itemd_key_per[0]=1
    for x in range(1,20):
        itemd_key_per[x]=np.int(np.percentile(itemd_keys,5*x))
    itemd_key_per[20]=itemd_keys[-1]+1
    itemd_prop=np.zeros(20)
    for x in range(20):
        itemd_prop[x]=len(list(filter(lambda y: itemd[y]>=itemd_key_per[x] and itemd[y]<itemd_key_per[x+1], range(num_items))))/num_items
    Cum=0
    for x in range(1,20):
        table.write(row+x,1,5*x)
        table.write(row+x,2,itemd_key_per[x])
        table.write(row+x,3,'{0}~{1}'.format(itemd_key_per[x-1], itemd_key_per[x]))
        table.write(row+x,4,itemd_prop[x-1])
        Cum+=itemd_prop[x-1]
        table.write(row+x,5,Cum)
        num=len(list(filter(lambda y: itemd[y]>itemd_key_per[x], range(num_items))))
        table.write(row+x,6,num)
    row+=22
    table.write(row,0,'calculating among item degrees')
    table.write(row,1,'percentile')
    table.write(row,2,'item degree')
    table.write(row,3,'number of items whose item degrees are larger than the given item degree')
    row+=1
    for x in range(9):
        p=(1+x)*10
        per=np.percentile(itemd,p)
        num=len(list(filter(lambda y: itemd[y]>per, range(num_items))))
        table.write(row,1,p)
        table.write(row,2,per)
        table.write(row,3,num)
        row+=1
    p=95
    per=np.percentile(itemd,p)
    num=len(list(filter(lambda x: itemd[x]>per, range(num_items))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=97.5
    per=np.percentile(itemd,p)
    num=len(list(filter(lambda x: itemd[x]>per, range(num_items))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=99
    per=np.percentile(itemd,p)
    num=len(list(filter(lambda x: itemd[x]>per, range(num_items))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=99.5
    per=np.percentile(itemd,p)
    num=len(list(filter(lambda x: itemd[x]>per, range(num_items))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=3

    #################针对于itemd_degree>itemd_keys的十分位数的项目进行如下处理##########################################################
    table.write(row,0,'keep items whose item degrees are larger than the {0}th percentile of itemd_keys'.format(split_item,))
    row+=1
    table.write(row,0,'remove users who have no ratings on remaining items')
    row+=1
    items=list(filter(lambda x: itemd[x]>np.percentile(itemd_keys,split_item), range(num_items)))
    users=np.array(Matrix_Index[:,items].sum(1).T)[0].nonzero()[0]
    Mat=Matrix[:,items][users]
    Mat_Ind=Matrix_Index[:,items][users]
    num_users,num_items=Mat.shape 
    table.write(row,0,'num_users')
    table.write(row,1,num_users)
    row+=1 
    table.write(row,0,'num_items')
    table.write(row,1,num_items)
    row+=1
    userd=np.array(Mat_Ind.sum(1).T)[0].astype(np.int)
    userd_keys=np.unique(userd)
    table.write(row,0,'calculating among the keys of user degrees')
    table.write(row,1,'percentile')
    table.write(row,2,'user degree')
    table.write(row,3,'user degree range')
    table.write(row,4,'percentage of users')
    table.write(row,5,'cumulative percentage')
    table.write(row,6,'number of user whose user degrees are larger than the given user degree')
    row+=1
    userd_key_per=[[]]*21
    userd_key_per[0]=1 
    for x in range(1,20):
        userd_key_per[x]=np.int(np.percentile(userd_keys, 5*x))
    userd_key_per[20]=userd_keys[-1]+1 
    userd_prop=np.zeros(20)
    for x in range(20):
        userd_prop[x]=len(list(filter(lambda y: userd[y]>=userd_key_per[x] and userd[y]<userd_key_per[x+1], range(num_users))))/num_users
    Cum=0
    for x in range(1,20):
        table.write(row,1,5*x)
        table.write(row,2,userd_key_per[x])
        table.write(row,3,'{0}~{1}'.format(userd_key_per[x-1], userd_key_per[x]))
        table.write(row,4,userd_prop[x-1])
        Cum+=userd_prop[x-1]
        table.write(row,5,Cum)
        num=len(list(filter(lambda y: userd[y]>userd_key_per[x], range(num_users))))
        table.write(row,6,num)
        row+=1 
    row+=2
    table.write(row,0,'calculating among user degrees')
    table.write(row,1,'percentile')
    table.write(row,2,'user degree')
    table.write(row,3,'number of users whose user degrees are larger than the given user degree')
    row+=1
    for x in range(9):
        p=(1+x)*10
        per=np.percentile(userd,p)
        num=len(list(filter(lambda y: userd[y]>per, range(num_users))))
        table.write(row,1,p)
        table.write(row,2,per)
        table.write(row,3,num)
        row+=1
    p=95
    per=np.percentile(userd,p)
    num=len(list(filter(lambda x: userd[x]>per, range(num_users))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=97.5
    per=np.percentile(userd,p)
    num=len(list(filter(lambda x: userd[x]>per, range(num_users))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=99
    per=np.percentile(userd,p)
    num=len(list(filter(lambda x: userd[x]>per, range(num_users))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=1
    p=99.5
    per=np.percentile(userd,p)
    num=len(list(filter(lambda x: userd[x]>per, range(num_users))))
    table.write(row,1,p)
    table.write(row,2,per)
    table.write(row,3,num)
    row+=3

    table.write(row,0,'Information for filtering data')
    row+=1
    table.write(row,0,'keep items whose item degrees are larger than the {0}th percentile of item_keys'.format(split_item,))
    row+=1
    table.write(row,0,'remove users who have no ratings on remaining items')
    row+=1
    table.write(row,0,'keep users whose user degrees are larger than the {0}th percentile of user_keys'.format(split_user,))
    row+=3

    users=list(filter(lambda x: userd[x]>np.percentile(userd_keys,split_user), range(num_users)))
    items=np.array(Mat_Ind[users].sum(0))[0].nonzero()[0]
    Final=Mat[users][:,items]
    Final_Ind=Mat_Ind[users][:,items]
    num_users,num_items=Final.shape 
    num_ratings=Final_Ind.sum()
    density=num_ratings/(num_users*num_items)
    aver_ratings=Final.sum()/num_ratings
    userd=np.array(Final_Ind.sum(1).T)[0]
    itemd=np.array(Final_Ind.sum(0))[0]
    table.write(row,0,'Final data information')
    table.write(row,1,'num_users')
    table.write(row,2, num_users)
    row+=1
    table.write(row,1,'num_items')
    table.write(row,2, num_items)
    row+=1 
    table.write(row,1,'num_ratings')
    table.write(row,2, num_ratings)
    row+=1 
    table.write(row,1,'density')
    table.write(row,2,density)
    row+=1
    table.write(row,1,'aver_ratings')
    table.write(row,2,aver_ratings)
    row+=1
    table.write(row,1,'user degree range')
    table.write(row,2, userd.min())
    table.write(row,3, userd.max())
    row+=1
    table.write(row,1,'item degree range')
    table.write(row,2, itemd.min())
    table.write(row,3, itemd.max())
    return Final,Final_Ind,table



    










res_file = xlwt.Workbook()
DataName_List=['AmazonBooks', 'MovieLens20M', 'Netflix']
split_pairs=[[10,15],[30,30],[50,30]]
tables=[res_file.add_sheet(name,cell_overwrite_ok=True) for name in DataName_List]
for table_ind,DataName in enumerate(DataName_List):
    table=tables[table_ind]
    split_user=split_pairs[table_ind][0]
    split_item=split_pairs[table_ind][1]
    Array=np.load('.\Data\{0}_Rating_Array.npy'.format(DataName,), allow_pickle=True)
    Matrix=np.load('.\Data\{0}_Rating_Matrix.npy'.format(DataName,), allow_pickle=True).tolist()
    num_users,num_items=Matrix.shape 
    num_ratings=len(Matrix.data)
    density=num_ratings/(num_users*num_items)
    table.write(0,0,'DataName')
    table.write(0,1,'{0}'.format(DataName,))
    table.write(1,0,'Initial Data')
    table.write(1,1,'num_users')
    table.write(1,2,num_users)
    table.write(2,1,'num_items')
    table.write(2,2,num_items)
    table.write(3,1,'num_ratings')
    table.write(3,2,num_ratings)
    table.write(4,1,'average_ratings')
    table.write(4,2,Matrix.sum()/num_ratings)
    table.write(5,1,'density')
    table.write(5,2,num_ratings/(num_users*num_items))
    indx,indy=Matrix.nonzero()
    Matrix_Index=scipy.sparse.csr_matrix((np.ones(len(indx)), (indx,indy)), shape=(num_users, num_items))
    row=6
    ###  Remove users who have only rated one items or rated all items only with one rating  #################
    if DataName=='AmazonBooks':
        Mat, Mat_Ind, table, row=filteringusers_1ratingsOr1items(Array, Matrix, Matrix_Index, table, row)
        #######  Filtering users and items accroding to their degrees  ###########################################
        Final_Mat, Final_Mat_Ind, table=filtering_user_items_percentile(Mat, Mat_Ind, table, row, split_user, split_item)
        
    else:
        #######  Filtering users and items accroding to their degrees  ###########################################
        Final_Mat, Final_Mat_Ind, table=filtering_user_items_percentile(Matrix, Matrix_Index, table, row, split_user, split_item) 
    OutPut={}
    OutPut['filtering item percentile']=split_item
    OutPut['filtering user percentile']=split_user
    OutPut['After Filtering: Matrix']=Final_Mat 
    OutPut['After Filtering: Matrix_Index']=Final_Mat_Ind
    OutPut['Before Filtering: Matrix']=Matrix 
    OutPut['Before Filtering: Matrix_Index']=Matrix_Index
    np.save('.\Data\Filtering_{0}_Results.npy'.format(DataName,),OutPut)




final_table=res_file.add_sheet('Filtering Data',cell_overwrite_ok=True)
titles=['DataName','num_users','num_items','num_ratings','density','rating_scales','aver ratings','aver user','aver item','item percentile','user percentile']
for col,title in enumerate(titles):
    final_table.write(0,col,title)

for row,DataName in enumerate(DataName_List):
    OutPut=np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()
    Matrix=OutPut['After Filtering: Matrix']
    Matrix_Index=OutPut['After Filtering: Matrix_Index']
    num_users,num_items=Matrix.shape 
    num_ratings=len(Matrix.data)
    density=num_ratings/(num_users*num_items)
    aver_ratings=Matrix.sum()/num_ratings
    rs=np.sort(np.unique(Matrix.data))
    userd=np.array(Matrix_Index.sum(1).T)[0]
    itemd=np.array(Matrix_Index.sum(0))[0]
    final_table.write(row+1,0,DataName)
    final_table.write(row+1,1,num_users)
    final_table.write(row+1,2,num_items)
    final_table.write(row+1,3,num_ratings)
    final_table.write(row+1,4,density)
    final_table.write(row+1,5,'{0}-{1}'.format(rs[0],rs[-1]))
    final_table.write(row+1,6,aver_ratings)
    final_table.write(row+1,7,'{0:.0f}({1},{2})'.format(userd.mean(), userd.min(), userd.max()))
    final_table.write(row+1,8,'{0:.0f}({1},{2})'.format(itemd.mean(), itemd.min(), itemd.max()))
    final_table.write(row+1,9,OutPut['filtering item percentile'])
    final_table.write(row+1,10,OutPut['filtering user percentile'])


res_file.save('.\Data\DataSet_Split_Statistics.xls')














