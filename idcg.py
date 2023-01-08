# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class Evaluation_Test(object):
    def __init__(self, Test_Matrix, Test_Mask, Test_Rank_Matrix, Test_Rank_Mask):
        self.Test_Matrix=Test_Matrix 
        self.Test_Mask=Test_Mask 
        self.Test_Rank_Matrix=Test_Rank_Matrix 
        self.Test_Rank_Mask=Test_Rank_Mask
        return
        
    def Ranking_Item(self, Matrix):
        ##According to the values in each row, ##
        ##present their specific location in the descending order of these values ##
        num_users,num_items=Matrix.shape 
        Sorted_Items=np.argsort(-1.0*Matrix, 1) # the value in (x,y) indicates a item's ID whose rating is ranked to be (y+1)th location in the x's unrated items#
        indx,indy=(Sorted_Items+1).nonzero()
        values=Sorted_Items[indx,indy]
        Item_Order=np.zeros((num_users, num_items))
        Item_Order[indx,values]=indy ##the value in position (x,y) displays the order of yth item among all items according to xth user's ratings.##
        return Item_Order
    
    def Evaluation_Rank(self):
        ####  Evaluate the rank of test items whose true ratings are >3.  ###################
        Test_Matrix=self.Test_Rank_Matrix 
        Test_Mask=self.Test_Rank_Mask
        ##Rank items according to their predicted ratings##
        Test_Matrix_RatingPow=np.power(2.0, Test_Matrix)-1.0
        Test_Item_TruOrder=self.Ranking_Item(Test_Matrix)
        idcg_denom=np.log2(Test_Item_TruOrder+2.0)
        idcg=np.true_divide(Test_Matrix_RatingPow, idcg_denom).sum(1) 
        return idcg

def Main_Function(TargetName, iteration):
    Val_Test=np.load('.\Data\{0}_Train_Val_Test_Data{1}.npy'.format(TargetName,iteration), allow_pickle=True).tolist()
    Test_Matrix=np.array(Val_Test['Test_Matrix'].todense())
    Test_Mask=np.array(Val_Test['Test_Mask'].todense())
    Val_Matrix=np.array(Val_Test['Val_Matrix'].todense())
    Val_Mask=np.array(Val_Test['Val_Mask'].todense())
    Test_Rank_Mask=np.array(Val_Test['Test_Rank_Mask'].todense())
    Test_Rank_Matrix=np.array(Val_Test['Test_Rank_Matrix'].todense())
    Val_Rank_Mask=np.array(Val_Test['Val_Rank_Mask'].todense())
    Val_Rank_Matrix=np.array(Val_Test['Val_Rank_Matrix'].todense())
    Eval_Val=Evaluation_Test(Val_Matrix, Val_Mask, Val_Rank_Matrix, Val_Rank_Mask)
    Eval_Test=Evaluation_Test(Test_Matrix, Test_Mask, Test_Rank_Matrix, Test_Rank_Mask)
    OutPut={}
    OutPut['TargetName']=TargetName 
    OutPut['iteration']=iteration 
    OutPut['Val: idcg']=Eval_Val.Evaluation_Rank()
    OutPut['Test: idcg']=Eval_Test.Evaluation_Rank()
    np.save('.\Data\{0}_idcg{1}.npy'.format(TargetName, iteration), OutPut)
    return 

#############################################################################
###############  Main Function  #############################################
#############################################################################
if __name__ == '__main__':
    TargetName_List=['AmazonBooks']
    for TargetName in TargetName_List:
        for iter in range(1):
            Main_Function(TargetName, iter)


