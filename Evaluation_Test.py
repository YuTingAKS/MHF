import numpy as np 
import scipy.sparse

class Evaluation_Test(object):
    def __init__(self, Train_Mask, Test_Matrix, Test_Mask, Test_Rank_Matrix, Test_Rank_Mask, idcg, Users_List):
        self.Train_Mask=Train_Mask 
        self.Test_Matrix=Test_Matrix 
        self.Test_Mask=Test_Mask 
        self.Test_Rank_Matrix=Test_Rank_Matrix 
        self.Test_Rank_Mask=Test_Rank_Mask
        self.Test_Rank_Matrix_RatingPow=np.power(2.0, Test_Rank_Matrix)-1.0
        self.Test_Rank_idcg=idcg
        self.Users_List=Users_List
        return
        
    def Evaluation_Pred(self, Pred):
        Test_Matrix=self.Test_Matrix
        Test_Mask=self.Test_Mask
        Users_List=self.Users_List
        ##Pred: unprocessed predicted ratings ##
        userind,itemind=Test_Matrix.nonzero()
        ratings=Test_Matrix[userind, itemind]
        num=len(ratings)
        pred_ratings=Pred[userind, itemind]
        ind_low=list(filter(lambda x: pred_ratings[x]<1.0, range(num)))
        ind_up=list(filter(lambda x: pred_ratings[x]>5.0, range(num)))
        pred_ratings[ind_low]=1.0 
        pred_ratings[ind_up]=5.0
        error=abs(pred_ratings-ratings)
        Error_Mat=np.zeros(Test_Matrix.shape)
        Error_Mat[userind, itemind]=error
        mae=np.zeros(len(Users_List))
        rmse=np.zeros(len(Users_List))
        for ind,users in enumerate(Users_List):
            if len(users)>0:
                Denom=Test_Mask[users].sum()
                mae[ind]=Error_Mat[users].sum()/Denom
                rmse[ind]=np.linalg.norm(Error_Mat[users])/np.sqrt(Denom)
        return mae, rmse
    
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
    
    def Evaluation_Rank45(self, Pred, topn):
        ####  Evaluate the rank of test items whose true ratings are >3.  ###################
        Test_Matrix=self.Test_Rank_Matrix 
        Test_Mask=self.Test_Rank_Mask
        Train_Mask=self.Train_Mask
        Test_Matrix_RatingPow=self.Test_Rank_Matrix_RatingPow
        idcg0=self.Test_Rank_idcg
        idcg=idcg0+np.equal(idcg0,0.0)
        Users_List=self.Users_List
        ##Pred: unprocessed predicted ratings ##
        userd=Test_Mask.sum(1)+np.equal(Test_Mask.sum(1),0.0)
        Pred_New=Pred-np.multiply(Train_Mask, Pred)+(Pred.min()-1.0)*Train_Mask ## Allocate the lowest ratings to users' rated items ##
        ##Rank items according to their predicted ratings##
        Item_Order=self.Ranking_Item(Pred_New)
        Test_Item_Order=np.multiply(Test_Mask, Item_Order) ## (num_users, num_items); if yth is a test item of xth user, (x,y) indicates the rank of this test item. ##
        denom=np.log2(Test_Item_Order+2.0)
        DCG_Mat=np.true_divide(Test_Matrix_RatingPow, denom) ## (2^r-1)/log_2(x+1) ##
        dcg=DCG_Mat.sum(1)
        Topn_Mask=np.multiply(np.less(Item_Order, topn), Test_Mask)## the value in position (x,y) indicates whether the test item y appears in user x's topn recommendation list ##
        tpn=Topn_Mask.sum(1)
        dcgn=np.multiply(DCG_Mat, Topn_Mask).sum(1)
        idcgn_numer=np.multiply(Test_Matrix_RatingPow, Topn_Mask)
        Topn_TruOrder=self.Ranking_Item(np.multiply(Test_Matrix, Topn_Mask))
        idcgn_denom=np.log2(Topn_TruOrder+2.0)
        idcgn0=np.true_divide(idcgn_numer, idcgn_denom).sum(1)
        idcgn=idcgn0+np.equal(idcgn0,0.0)

        ndcg=np.zeros(len(Users_List))
        pre=np.zeros(len(Users_List))
        re=np.zeros(len(Users_List))
        ndcgn=np.zeros(len(Users_List))
        for ind,users in enumerate(Users_List):
            if len(users)>0:
                ndcg[ind]=np.true_divide(dcg[users], idcg[users]).mean()
                pre[ind]=(tpn[users]/topn).mean()
                re[ind]=np.true_divide(tpn[users], userd[users]).mean()
                ndcgn[ind]=np.true_divide(dcgn[users], idcgn[users]).mean()
        return ndcg,pre,re,ndcgn


def Evaluation_Function(OutPut, Pred_List, PredKeys_List, \
    Train_Mask, Val_Matrix, Val_Rank_Matrix, Val_Mask, Val_Rank_Mask, Test_Matrix, Test_Rank_Matrix, Test_Mask, Test_Rank_Mask, \
        val_idcg, test_idcg, Users_List, UserKeys_List, topN):
    for pred_ind in range(len(Pred_List)):
        Pred=Pred_List[pred_ind]
        PredKeys=PredKeys_List[pred_ind]
        # Eval_Val=Evaluation_Test(Train_Mask, Val_Matrix, Val_Mask, Val_Rank_Matrix, Val_Rank_Mask, val_idcg, Users_List)
        # MAE1,RMSE1=Eval_Val.Evaluation_Pred(Pred)
        # NDCG1,Pre1,Re1,NDCGN1=Eval_Val.Evaluation_Rank45(Pred, topN)
        # for ind1,UserKeys in enumerate(UserKeys_List):
        #     OutPut[PredKeys+'Val: mae'+UserKeys]=MAE1[ind1]
        #     OutPut[PredKeys+'Val: rmse'+UserKeys]=RMSE1[ind1]
        #     OutPut[PredKeys+'Val Part: ndcg'+UserKeys]=NDCG1[ind1]
        #     OutPut[PredKeys+'Val Part: precision@{0}'.format(topN,)+UserKeys]=Pre1[ind1]
        #     OutPut[PredKeys+'Val Part: recall@{0}'.format(topN,)+UserKeys]=Re1[ind1]
        #     OutPut[PredKeys+'Val Part: ndcg@{0}'.format(topN,)+UserKeys]=NDCGN1[ind1]
        Eval_Test=Evaluation_Test(Train_Mask+Val_Mask, Test_Matrix, Test_Mask, Test_Rank_Matrix, Test_Rank_Mask, test_idcg, Users_List)
        MAE2,RMSE2=Eval_Test.Evaluation_Pred(Pred)
        NDCG2,Pre2,Re2,NDCGN2=Eval_Test.Evaluation_Rank45(Pred, topN)
        for ind2,UserKeys in enumerate(UserKeys_List):
            OutPut[PredKeys+'Test: mae'+UserKeys]=MAE2[ind2]
            OutPut[PredKeys+'Test: rmse'+UserKeys]=RMSE2[ind2]
            OutPut[PredKeys+'Test Part: ndcg'+UserKeys]=NDCG2[ind2]
            OutPut[PredKeys+'Test Part: precision@{0}'.format(topN,)+UserKeys]=Pre2[ind2]
            OutPut[PredKeys+'Test Part: recall@{0}'.format(topN,)+UserKeys]=Re2[ind2]
            OutPut[PredKeys+'Test Part: ndcg@{0}'.format(topN,)+UserKeys]=NDCGN2[ind2]
    return OutPut

