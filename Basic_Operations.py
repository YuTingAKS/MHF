import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def Validation_Rank_GPU(Pred, Train_Mask, Val_Mask, Rating_Pow, Denom_Log, val_idcg, val_userd_float):
    num_items=Train_Mask.shape[1]
    ##num_items: constant in cpu ##
    a=tf.constant(2, dtype=tf.float32)
    Pred_New=Pred-tf.multiply(Pred, Train_Mask)+(tf.reduce_min(Pred)-1.0)*Train_Mask
    Sorted_Items=tf.nn.top_k(Pred_New, num_items)[1] #### Ranking unrated items of a user in descending order of predicted ratings#######################
    del Pred_New
    Item_Order=tf.nn.top_k(-1*Sorted_Items, num_items)[1] ##the value in position (x,y) displays the order of yth item among all items according to xth user's predicted ratings.##
    del Sorted_Items
    Test_Item_Order=tf.multiply(tf.cast(Item_Order, dtype=tf.float32), Val_Mask) 
    denom=tf.truediv(tf.math.log(Test_Item_Order+a), tf.math.log(a))
    del Test_Item_Order
    DCG=tf.truediv(Rating_Pow, denom)
    dcg=tf.reduce_sum(DCG, 1)
    NDCG=tf.reduce_mean(tf.truediv(dcg,val_idcg))
    del denom, dcg
    Recom20_Mask=tf.multiply(Val_Mask, tf.cast(tf.less(Item_Order,20), dtype=tf.float32)) ### (num_users, num_items); Whether do test items exist in topn_list ### 
    del Item_Order
    TP20=tf.reduce_sum(Recom20_Mask,1)
    Precision20=tf.reduce_mean(tf.truediv(TP20, 20.0))
    Recall20=tf.reduce_mean(tf.truediv(TP20, val_userd_float))
    Recom20_DCG=tf.reduce_sum(tf.multiply(DCG,Recom20_Mask),1)
    del DCG
    Recom20_Rating_Pow=tf.multiply(Rating_Pow, Recom20_Mask)
    del Recom20_Mask
    tp20_max=tf.cast(tf.reduce_max(TP20), dtype=tf.int32) 
    Recom20_IDCG=tf.reduce_sum(tf.truediv(tf.nn.top_k(Recom20_Rating_Pow, tp20_max)[0], Denom_Log[:tp20_max]),1)
    del Recom20_Rating_Pow, TP20
    NDCG20=tf.reduce_mean(tf.truediv(Recom20_DCG, Recom20_IDCG+tf.cast(tf.equal(Recom20_IDCG,0.0), dtype=tf.float32))) 
    del Recom20_DCG, Recom20_IDCG
    return NDCG,Precision20,Recall20,NDCG20

def Validation_GPU(Pred, Train_Mask, Val_Matrix, Val_Mask, Rating_Pow, Denom_Log, val_userd_float):
    ##num_items: constant in cpu ##
    num_items=Pred.shape[1]
    a=tf.constant(2, dtype=tf.float32)
    RMSE=tf.sqrt(tf.truediv(tf.reduce_sum(tf.squared_difference(tf.multiply(Pred, Val_Mask),Val_Matrix)), \
        tf.reduce_sum(Val_Mask)))
    Pred_New=Pred-tf.multiply(Pred, Train_Mask)+(tf.reduce_min(Pred)-1.0)*Train_Mask
    Sorted_Items=tf.nn.top_k(Pred_New, num_items)[1] #### Ranking unrated items of a user in descending order of predicted ratings#######################
    Item_Order=tf.nn.top_k(-1*Sorted_Items, num_items)[1] ##the value in position (x,y) displays the order of yth item among all items according to xth user's predicted ratings.##
    Test_Item_Order=tf.multiply(tf.cast(Item_Order, dtype=tf.float32), Val_Mask) 
    denom=tf.truediv(tf.log(Test_Item_Order+a), tf.log(a))
    DCG=tf.truediv(Rating_Pow, denom)
    Recom20_Mask=tf.multiply(Val_Mask, tf.cast(tf.less(Item_Order,20), dtype=tf.float32)) ### (num_users, num_items); Whether do test items exist in topn_list ### 
    TP20=tf.reduce_sum(Recom20_Mask,1)
    Precision20=tf.reduce_mean(tf.truediv(TP20, 20.0))
    Recall20=tf.reduce_mean(tf.truediv(TP20, val_userd_float))
    Recom20_DCG=tf.reduce_sum(tf.multiply(DCG,Recom20_Mask),1)
    Recom20_Rating_Pow=tf.multiply(Rating_Pow, Recom20_Mask)
    tp20_max=tf.cast(tf.reduce_max(TP20), dtype=tf.int32) 
    Recom20_IDCG=tf.reduce_sum(tf.truediv(tf.nn.top_k(Recom20_Rating_Pow, tp20_max)[0], Denom_Log[:tp20_max]),1)
    NDCG20=tf.reduce_mean(tf.truediv(Recom20_DCG, Recom20_IDCG+tf.cast(tf.equal(Recom20_IDCG,0.0), dtype=tf.float32))) 
    return RMSE,Precision20,Recall20,NDCG20,

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def Khatri_Rao_GPU(A, B):
    ## Calculating A@B ##
    C1=tf.reshape(tf.tile(A, (1, B.shape[0])), shape=(-1, A.shape[1])) ###(rowa, cola*rowb)==> (rowa*rowb, cola)####
    C2=tf.tile(B, (A.shape[0], 1)) ## (rowb*rowa, colb) ##
    C=tf.multiply(C1, C2)
    del C1, C2
    return C

def Tensor2Matrix1_GPU(T):
    ###T:(K,I,J)##
    T1=tf.reshape(tf.transpose(T, [1,0,2]), shape=[T.shape[1], -1]) 
    return T1 

def Tensor2Matrix2_GPU(T):
    ###T:(K,I,J)##
    T2=tf.reshape(tf.transpose(T, [2,0,1]), shape=[T.shape[2], -1]) 
    return T2 

def Tensor2Matrix3_GPU(T):
    ###T:(K,I,J)##
    T3=tf.reshape(tf.transpose(T, [0,2,1]), shape=[T.shape[0], -1]) 
    return T3 

def Calculating_IDCG(Matrix, Mask, rated_items):
    num_users=Matrix.shape[0]
    userd=Mask.sum(1)
    ratings=list(map(lambda u: Matrix[u, rated_items[u]], range(num_users)))
    indices=list(map(lambda u: np.argsort(-ratings[u]), range(num_users))) ## Ranking items in descending order of ratings ##
    sorted_ratings=list(map(lambda u: ratings[u][indices[u]], range(num_users)))
    Numer=list(map(lambda u: np.power(2, sorted_ratings[u])-1, range(num_users)))
    Denom=list(map(lambda u: np.log2(np.arange(userd[u])+2), range(num_users)))
    IDCG=np.array(list(map(lambda u: np.true_divide(Numer[u], Denom[u]).sum(), range(num_users))))
    return IDCG  