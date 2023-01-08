import numpy as np
import random
# import tensorflow as tf
from copy import deepcopy
import scipy.sparse
import Basic_Operations
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class CodeBook_Construction(object):
    def __init__(self, Matrix, num_user_clusters, num_item_clusters, rating_scales=np.arange(1,6), max_epoches=1000, seed=42, tol=1e-2):
        self.Matrix=Matrix 
        self.num_user_clusters,self.num_item_clusters=num_user_clusters,num_item_clusters 
        self.max_epoches=max_epoches 
        self.seed=seed 
        self.tol=tol 
        self.num_users,self.num_items=Matrix.shape 
        self.Mask=np.zeros((self.num_users, self.num_items))
        self.rating_scales=rating_scales
        userind,itemind=Matrix.nonzero()
        self.Mask[userind, itemind]=1 
        np.random.seed(seed)
        U=np.zeros((self.num_users, num_user_clusters))
        U[np.arange(self.num_users),np.random.choice(np.arange(num_user_clusters), self.num_users)]=1
        self.U0=np.true_divide(U,np.sqrt(np.power(U,2).sum(0)))
        np.random.seed(seed)
        V=np.zeros((self.num_items, num_item_clusters))
        V[np.arange(self.num_items),np.random.choice(np.arange(num_item_clusters), self.num_items)]=1
        self.V0=np.true_divide(V,np.sqrt(np.power(V,2).sum(0)))
        np.random.seed(seed)
        self.S0=np.random.choice(rating_scales, num_user_clusters*num_item_clusters).reshape(-1,num_item_clusters)
    
    def reset_graph(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
    
    def Generating_CodeBook(self):
        self.ONMTF()
        U_C=np.zeros((self.num_users, self.num_user_clusters))
        V_C=np.zeros((self.num_items, self.num_item_clusters))
        indUx=np.arange(self.num_users)
        indUy=np.array(list(map(lambda u: np.argmax(self.U[u]), range(self.num_users))))
        indVx=np.arange(self.num_items)
        indVy=np.array(list(map(lambda v: np.argmax(self.V[v]), range(self.num_items))))
        U_C[indUx,indUy]=1
        V_C[indVx,indVy]=1
        Numer=U_C.T.dot(self.Matrix).dot(V_C)
        Denom=U_C.T.dot(self.Mask).dot(V_C)
        CodeBook=np.true_divide(Numer, Denom+np.equal(Denom,0.0))
        self.U_C=U_C 
        self.V_C=V_C 
        self.CodeBook=CodeBook
        return CodeBook

    def Complementing_Missing_Values(self):
        RM=deepcopy(self.Matrix)
        userd=self.Mask.sum(1) ##number of items rated by each users##
        gmeans=RM.sum()/userd.sum()  ##global means##
        means=np.array(list(map(lambda u: RM[u].sum()/userd[u] if userd[u]>0 else gmeans, range(self.num_users)))) 
        indx,indy=(1-self.Mask).nonzero()
        RM[indx,indy]=means[indx]
        Completed_Matrix=RM 
        return Completed_Matrix
    
    def ONMTF(self):
        RM=self.Complementing_Missing_Values() ###Completed rating matrix with the mean values of users#####
        self.reset_graph()
        X=tf.constant(RM, dtype=tf.float32)
        U0=tf.constant(self.U0, dtype=tf.float32)
        V0=tf.constant(self.V0, dtype=tf.float32)
        S0=tf.constant(self.S0, dtype=tf.float32)
        # U0=tf.random_uniform(shape=[self.num_users, self.num_user_clusters], minval=0, maxval=1, dtype=tf.float32, seed=self.seed)
        # V0=tf.random_uniform(shape=[self.num_items, self.num_item_clusters], minval=0, maxval=1, dtype=tf.float32, seed=self.seed)
        # S0=tf.random.uniform(shape=[self.num_user_clusters, self.num_item_clusters], minval=0, maxval=1, dtype=tf.float32, seed=self.seed)
        tol=tf.constant(self.tol, dtype=tf.float32)
        ## Updating V,U,S once  #########################
        V_numer = tf.matmul(X, tf.matmul(U0, S0), transpose_a=True)
        V_denom = tf.matmul(V0, tf.matmul(V0, V_numer, transpose_a=True))
        V = tf.multiply(V0, tf.pow(tf.truediv(V_numer, V_denom), 0.5))
        del V_numer,V_denom
        U_numer = tf.matmul(X, tf.matmul(V, S0, transpose_b=True))
        U_denom = tf.matmul(U0, tf.matmul(U0, U_numer, transpose_a=True))
        U = tf.multiply(U0, tf.pow(tf.truediv(U_numer, U_denom), 0.5))
        del U_numer,U_denom
        S_numer = tf.matmul(U, tf.matmul(X, V), transpose_a=True)
        UTU = tf.matmul(U, U, transpose_a=True)
        VTV = tf.matmul(V, V, transpose_a=True)
        S_denom = tf.matmul(UTU, tf.matmul(S0, VTV))
        S = tf.multiply(S0, tf.pow(tf.truediv(S_numer, S_denom), 0.5))
        del S_numer,S_denom,UTU,VTV
        ## Calculating loss and updating X ##
        Pred = tf.matmul(U, tf.matmul(S, V, transpose_b=True))
        Error=tf.nn.l2_loss(tf.subtract(X, Pred))
        diff=10+tol 
        Error_Array=tf.reshape(tf.concat([[Error], [diff]], axis=0), shape=[1,-1])
        del Pred,Error

        def cond(diff, U, S, V, U0, S0, V0, Error_Array):
            return diff > tol

        def body(diff, U, S, V, U0, S0, V0, Error_Array):
            del U0, S0, V0, diff
            ##Updating U0,S0,V0##
            U0=U 
            S0=S 
            V0=V
            ## Udating V,U,S once ##
            V_numer = tf.matmul(X, tf.matmul(U, S), transpose_a=True)
            V_denom = tf.matmul(V, tf.matmul(V, V_numer, transpose_a=True))
            V = tf.multiply(V, tf.pow(tf.truediv(V_numer, V_denom), 0.5))
            del V_numer,V_denom
            U_numer = tf.matmul(X, tf.matmul(V, S, transpose_b=True))
            U_denom = tf.matmul(U, tf.matmul(U, U_numer, transpose_a=True))
            U = tf.multiply(U, tf.pow(tf.truediv(U_numer, U_denom), 0.5))
            del U_numer,U_denom
            S_numer = tf.matmul(U, tf.matmul(X, V), transpose_a=True)
            UTU = tf.matmul(U, U, transpose_a=True)
            VTV = tf.matmul(V, V, transpose_a=True)
            S_denom = tf.matmul(UTU, tf.matmul(S, VTV))
            S = tf.multiply(S, tf.pow(tf.truediv(S_numer, S_denom), 0.5))
            del S_numer,S_denom,UTU,VTV
            Pred = tf.matmul(U, tf.matmul(S, V, transpose_b=True))
            Error=tf.nn.l2_loss(tf.subtract(X, Pred))
            diff=Error_Array[-1,0]-Error 
            EE=tf.reshape(tf.concat([[Error], [diff]], axis=0), shape=[1,-1])
            Error_Array=tf.concat([Error_Array, EE], axis=0)
            del Pred,Error,EE
            return [diff, U, S, V, U0, S0, V0, Error_Array]
        
        _, _, _, _, U02, S02, V02, Error_Array2 = tf.while_loop(cond, body, loop_vars=[diff, U, S, V, U0, S0, V0, Error_Array],\
            shape_invariants=[diff.get_shape(), U.get_shape(), S.get_shape(), V.get_shape(), \
                U0.get_shape(), S0.get_shape(), V0.get_shape(), tf.TensorShape([None, 2])],\
                    maximum_iterations=self.max_epoches)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # print('No OK')
            # start = time.clock()
            U_New, S_New, V_New, Error_Array_New=sess.run([U02,S02,V02,Error_Array2])
            # print(time.clock()-start)
            # print('OK')
        self.U=U_New 
        self.S=S_New 
        self.V=V_New
        self.Error_Array=Error_Array_New
        return


class FMM_Train_TF(object):
    def __init__(self, Train_Matrix, PCu0, PCv0, Pr0, num_user_clusters, num_item_clusters, rating_scales, split, ratio=0.2, max_epoches=200, seed=42, tol=1e-2):
        ### DataName_List: (target domain, source domain 1, source domain 2, ........) ###
        self.Train_Matrix=Train_Matrix 
        self.num_users, self.num_items=Train_Matrix.shape 
        self.num_user_clusters,self.num_item_clusters=num_user_clusters, num_item_clusters
        self.rating_scales=rating_scales
        self.max_epoches,self.seed,self.tol=max_epoches, seed, tol
        self.ratio=ratio
        self.split=split
        R=len(rating_scales)
        self.R=R
        userind,itemind=Train_Matrix.nonzero()
        ratings=Train_Matrix[userind, itemind]
        self.Train_Array=np.vstack([userind, itemind, ratings]).T 
        self.PCu0,self.PCv0,self.Pr0=PCu0,PCv0,Pr0
        np.random.seed(seed)
        PUC0=np.random.uniform(size=[self.num_users, self.num_user_clusters])
        denom0=PUC0.sum(0)
        denom=denom0+np.equal(denom0,0.0)
        self.PUC0=np.true_divide(PUC0, denom)
        np.random.seed(seed)
        PVC0=np.random.uniform(size=[self.num_items, self.num_item_clusters])
        denom0=PVC0.sum(0)
        denom=denom0+np.equal(denom0,0.0)
        self.PVC0=np.true_divide(PVC0, denom) 
        return
    
    def Main_Function(self):
        ######    AEM    ###################################
        Train1_Array, Train2_Array=self.Data_Train_Split()
        self.train1_interval,self.train1_userind,self.train1_itemind,self.train1_ratings,self.train1_ratingind=self.Data_Processing(Train1_Array)
        self.train2_interval,self.train2_userind,self.train2_itemind,self.train2_ratings,self.train2_ratingind=self.Data_Processing(Train2_Array)
        ######    EM    ###################################
        self.train_interval,self.train_userind,self.train_itemind,self.train_ratings,self.train_ratingind=self.Data_Processing(self.Train_Array)
        Basic_Operations.reset_graph()
        tol=tf.constant(self.tol, dtype=tf.float32)
        rating_scales=tf.constant(np.array(self.rating_scales).reshape(-1,1), dtype=tf.float32) ## (R,1) ##
        # Train_Array=tf.constant(self.Train_Array)
        #####  Anneal EM to produce possibility and b#########################################
        PCu0=tf.constant(self.PCu0, dtype=tf.float32)
        PCv0=tf.constant(self.PCv0, dtype=tf.float32)
        Pr0=tf.constant(self.Pr0, dtype=tf.float32)
        PUC0=tf.constant(self.PUC0, dtype=tf.float32)
        PVC0=tf.constant(self.PVC0, dtype=tf.float32)
        PCu1,PCv1,Pr1,PUC1,PVC1,b1,error_array1=self.Training_AEM(PCu0,PCv0,Pr0,PUC0,PVC0,rating_scales,tol)
        #####  EM  #############
        PCu2,PCv2,Pr2,PUC2,PVC2,error_array2=self.Training_EM(PCu1,PCv1,Pr1,PUC1,PVC1,b1,rating_scales,tol)
        init = tf.global_variables_initializer() 
        with tf.Session() as sess:
            sess.run(init)
            aem_pcu1,aem_pcv1,aem_pr1,aem_puc1,aem_pvc1,aem_b,aem_loss,em_pcu2,em_pcv2,em_pr2,em_puc2,em_pvc2,em_loss=sess.run(\
                [PCu1,PCv1,Pr1,PUC1,PVC1,b1,error_array1,PCu2,PCv2,Pr2,PUC2,PVC2,error_array2])
            self.AEM_PCu,self.AEM_PCv,self.AEM_Pr,self.AEM_PUC,self.AEM_PVC=aem_pcu1,aem_pcv1,aem_pr1,aem_puc1,aem_pvc1
            self.AEM_b,self.AEM_Loss=aem_b,aem_loss 
            self.EM_PCu,self.EM_PCv,self.EM_Pr,self.EM_PUC,self.EM_PVC=em_pcu2,em_pcv2,em_pr2,em_puc2,em_pvc2
            self.EM_Loss=em_loss
        return 

    def Data_Train_Split(self):
        Train_Array=self.Train_Array
        num_ratings=len(Train_Array)
        num=np.int(np.ceil(self.ratio*num_ratings))
        random.seed(self.seed)
        ind=random.sample(list(np.arange(num_ratings)), num)
        Train2_Array=Train_Array[ind]
        Train1_Array=np.delete(Train_Array, ind, 0)
        self.Train1_Array=Train1_Array 
        self.Train2_Array=Train2_Array
        return Train1_Array, Train2_Array  
    
    def Data_Processing(self,Train_Data):
        ##Dividing training users into separated parts##
        Num=len(Train_Data)
        split_num=Num//self.split+1-np.equal(Num%self.split,0)
        interval=[x*self.split for x in range(split_num+1)]
        interval[-1]=Num 
        userind=Train_Data[:,0].astype(np.int)
        itemind=Train_Data[:,1].astype(np.int) 
        ratings=Train_Data[:,2]
        ratingind=np.array(list(map(lambda x: self.rating_scales.index(x), ratings)))
        return interval, userind, itemind, ratings, ratingind
    
    def EMCalculating_QDenom(self, P_User, P_Item, Pr, userind, itemind, ratingind, b):
        K=tf.shape(P_User)[1]
        L=tf.shape(P_Item)[1]
        N=tf.shape(userind)[0]
        ind=tf.constant(0, dtype=tf.int32)
        Denom=tf.zeros(N, dtype=tf.float32)
        def cond(ind, Denom):
            return ind<K 
        def body(ind, Denom):
            PU=tf.reshape(P_User[:,ind], shape=[-1,1]) ## (Nu,1) ##
            PPU=tf.tile(tf.gather_nd(PU, userind), [1,L]) ## (N,L) ##
            PPV=tf.gather_nd(P_Item, itemind) ## (N,L) ##
            PPr=tf.gather_nd(Pr[:,ind,:], ratingind) ## (N,L) ##
            Numer=tf.pow(tf.multiply(tf.multiply(PPU,PPr),PPV), b) ## (N,L) ##
            Denom+=tf.reduce_sum(Numer,1) ## (N,) ##
            ind+=1 
            del PU, PPU, PPV, PPr, Numer
            return [ind, Denom]
        _,Q_Denom=tf.while_loop(cond, body, loop_vars=[ind,Denom], \
            shape_invariants=[ind.get_shape(), Denom.get_shape()], maximum_iterations=K)
        return Q_Denom
    
    def EMCalculating_Numer(self, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer,\
        P_User, P_Item, Pr, Q_Denom, userind, itemind, ratingind, b):
        Nu=tf.shape(P_User)[0]
        Ni=tf.shape(P_Item)[0]
        R=tf.shape(Pr)[0]
        K=tf.shape(P_User)[1]
        L=tf.shape(P_Item)[1]
        N=tf.shape(userind)[0]
        values=tf.ones(N, dtype=tf.float32)
        yind=tf.range(tf.cast(N, dtype=tf.int64), dtype=tf.int64)
        indices=tf.stack([tf.reshape(userind, shape=[-1,]), yind], axis=1)
        UserInd_Mat=tf.sparse_reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=[Nu, N]))
        del indices
        indices=tf.stack([tf.reshape(itemind, shape=[-1,]), yind], axis=1)
        ItemInd_Mat=tf.sparse_reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=[Ni, N]))
        del indices
        indices=tf.stack([tf.reshape(ratingind, shape=[-1,]), yind], axis=1)
        RatingInd_Mat=tf.sparse_reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=[R, N]))
        del indices, yind, values 
        ind=tf.constant(0, dtype=tf.int32)
        QD_Ind=tf.cast(tf.equal(Q_Denom, 0.0), dtype=tf.float32) ## (N,) ##
        Q_Denom+=QD_Ind
        Q2=tf.tile(tf.reshape(tf.cast(tf.truediv(1, K*L), dtype=tf.float32)*QD_Ind, shape=[-1,1]), [1,L])## (N, L) ##
        del QD_Ind
        def cond(ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer):
            return ind<K 
        def body(ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer):
            PU=tf.reshape(P_User[:,ind], shape=[-1,1]) ## (Nu,1) ##
            PPU=tf.tile(tf.gather_nd(PU, userind), [1,L]) ## (N,L) ##
            PPV=tf.gather_nd(P_Item, itemind) ## (N,L) ##
            PPr=tf.gather_nd(Pr[:,ind,:], ratingind) ## (N,L) ##
            Numer=tf.pow(tf.multiply(tf.multiply(PPU,PPr),PPV), b) ## (N,L) ##
            Q2d=tf.transpose(tf.truediv(tf.transpose(Numer), Q_Denom))+Q2 ## (N,L) ##
            del PU, PPU, PPV, PPr, Numer
            part1=PCu_Numer[:ind]
            part2=tf.reshape(PCu_Numer[ind]+tf.reduce_sum(Q2d),shape=[-1,])
            part3=PCu_Numer[(ind+1):]
            PCu_Numer=tf.concat([part1, part2, part3], axis=0) ## (K,) ##
            del part1, part2, part3 
            PCv_Numer+=tf.reduce_sum(Q2d, 0) ## (L,) ##
            part1=PUC_Numer[:,:ind]
            part2=tf.reshape(PUC_Numer[:,ind], shape=[-1,1])+tf.sparse_tensor_dense_matmul(UserInd_Mat, tf.reshape(tf.reduce_sum(Q2d, 1), shape=[-1,1])) ## (Nu,1) ##
            part3=PUC_Numer[:,(ind+1):]
            PUC_Numer=tf.concat([part1, part2, part3], axis=1) ## (Nu, K) ##
            del part1, part2, part3
            PVC_Numer+=tf.sparse_tensor_dense_matmul(ItemInd_Mat, Q2d) ## (Ni, L) ##
            part1=Pr_Numer[:,:(ind*L)]
            part2=Pr_Numer[:,(ind*L):((ind+1)*L)]+tf.sparse_tensor_dense_matmul(RatingInd_Mat, Q2d) ## (R, L) ##
            part3=Pr_Numer[:,((ind+1)*L):]
            Pr_Numer=tf.concat([part1, part2, part3], axis=1) ## (R, KL) ##
            del part1, part2, part3
            ind+=1 
            del Q2d
            return [ind,PCu_Numer,PCv_Numer,PUC_Numer,PVC_Numer,Pr_Numer]
        _,PCu_Numer2,PCv_Numer2,PUC_Numer2,PVC_Numer2,Pr_Numer2=tf.while_loop(cond, body, \
            loop_vars=[ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer],\
                shape_invariants=[ind.get_shape(), tf.TensorShape([None,]), tf.TensorShape([None,]), \
                    tf.TensorShape([None,None]), tf.TensorShape([None,None]), tf.TensorShape([None,None])],\
                        maximum_iterations=K)
        del UserInd_Mat, ItemInd_Mat, RatingInd_Mat, Q2
        return PCu_Numer2,PCv_Numer2,PUC_Numer2,PVC_Numer2,Pr_Numer2

    def Expectation_Maximization_TF(self, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer, \
        P_User, P_Item, Pr, interval, userind, itemind, ratingind, b):
        ##P_User   : (Nu, K)##
        ##P_Item   : (Ni, L)##
        ##Pr       : (R,K,L)##
        ##userind  : (N,1)##
        ##itemind  : (N,1)##
        ##ratingind: (N,1)##
        split_num=tf.shape(interval)[0]-1
        RKL=tf.shape(Pr)
        K=RKL[1]
        L=RKL[2]
        # Nu=tf.shape(P_User)[0]
        # Ni=tf.shape(P_Item)[0]
        N=tf.shape(userind)[0]
        b=tf.reshape(b, shape=[-1,])
        #### Calculating the Denomerator of Q ####
        ind=tf.constant(0, dtype=tf.int32)
        def cond(ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer):
            return ind<split_num
        def body(ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer):
            start=tf.gather_nd(interval, [ind,])
            end=tf.gather_nd(interval, [ind+1,])
            num=end-start
            users=tf.slice(userind, [start,0], [num,1]) ## (num,1) ##
            items=tf.slice(itemind, [start,0], [num,1]) ## (num,1) ##
            indices=tf.slice(ratingind, [start,0], [num,1]) ## (num,1) ##
            Denom=self.EMCalculating_QDenom(P_User, P_Item, Pr, users, items, indices, b) ## (num,) ##
            PCu_Numer,PCv_Numer,PUC_Numer,PVC_Numer,Pr_Numer=self.EMCalculating_Numer(\
                PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer,\
                    P_User, P_Item, Pr, Denom, users, items, indices, b)
            ind+=1
            del Denom
            return [ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer]
        _,PCu_Numer_All,PCv_Numer_All,PUC_Numer_All,PVC_Numer_All,Pr_Numer_All=tf.while_loop(cond, body, \
            loop_vars=[ind, PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer],\
                shape_invariants=[ind.get_shape(), tf.TensorShape([None,]), tf.TensorShape([None,]), \
                    tf.TensorShape([None,None]), tf.TensorShape([None,None]), tf.TensorShape([None,None])],\
                        maximum_iterations=split_num)
        del P_User, P_Item
        NN=tf.cast(N, dtype=tf.float32)
        PCu=tf.truediv(PCu_Numer_All, NN) ## (K,) ##
        PCv=tf.truediv(PCv_Numer_All, NN) ## (L,) ##
        del PCu_Numer_All, PCv_Numer_All
        denom=NN*PCu+tf.cast(tf.equal(NN*PCu, 0.0), dtype=tf.float32)
        PUC=tf.truediv(PUC_Numer_All, denom) ## (Nu, K) ##
        del denom, PUC_Numer_All
        denom=NN*PCv+tf.cast(tf.equal(NN*PCv, 0.0), dtype=tf.float32)
        PVC=tf.truediv(PVC_Numer_All, denom) ## (Ni, L) ##
        del denom, PVC_Numer_All
        denom=tf.reduce_sum(Pr_Numer_All, 0)+tf.cast(tf.equal(tf.reduce_sum(Pr_Numer_All, 0), 0.0), dtype=tf.float32)
        Pr=tf.reshape(tf.truediv(Pr_Numer_All, denom), shape=[-1,K,L])
        del denom, Pr_Numer_All
        # NN=tf.cast(N, dtype=tf.float32)
        # PCu=tf.truediv(PCu_Numer_All, NN) ## (K,) ##
        # PCv=tf.truediv(PCv_Numer_All, NN) ## (L,) ##
        # PUC=tf.truediv(PUC_Numer_All, NN*PCu) ## (Nu, K) ##
        # PVC=tf.truediv(PVC_Numer_All, NN*PCv) ## (Ni, L) ##
        # Pr=tf.reshape(tf.truediv(Pr_Numer_All, tf.reduce_sum(Pr_Numer_All, 0)), shape=[-1,K,L])
        return PCu, PCv, PUC, PVC, Pr
    
    def Prediction_TF(self,P_User, P_Item, Pr, userind, itemind, rating_scales):
        ##P_User       : a (Nu, K) array; P(u,Cu)##
        ##P_Item       : a (Ni, L) array; P(v,Cv)##
        ##Pr           : a (R, K*L) array; P(r|Cu,Cv)##
        ##userind      : a (N2, 1) array##
        ##itemind      : a (N2, 1) array##
        ##rating_scales: a (R, 1) array##
        K=tf.shape(P_User)[1]
        L=tf.shape(P_Item)[1]
        N=tf.shape(userind)[0]
        R=tf.shape(Pr)[0]
        Q=tf.zeros([N,R], dtype=tf.float32)
        ind=tf.constant(0, dtype=tf.int32)
        def cond(ind, Q):
            return ind<K 
        def body(ind,Q):
            PU=tf.reshape(P_User[:,ind], shape=[-1,1]) ## (Nu,1) ##
            PPU=tf.tile(tf.gather_nd(PU, userind), [1,L]) ## (N,L) ##
            PPV=tf.gather_nd(P_Item, itemind) ## (N,L) ##
            PPr=tf.transpose(Pr[:,ind,:]) ## (L,R) ##
            Q+=tf.matmul(tf.multiply(PPU, PPV), PPr) ## (N,R) ##
            ind+=1
            del PU, PPU, PPV, PPr
            return [ind,Q]
        _,Q2=tf.while_loop(cond, body, loop_vars=[ind, Q],\
            shape_invariants=[ind.get_shape(), Q.get_shape()],maximum_iterations=K)
        Numer=tf.reshape(tf.matmul(Q2,rating_scales), shape=[-1,]) ## (N,) ##
        Denom0=tf.reduce_sum(Q2, 1) ## (N,) ##
        del Q2
        Denom1=tf.cast(tf.equal(Denom0, 0), dtype=tf.float32) ## (N,) ##
        Denom=tf.add(Denom0,Denom1) ## (N,) ##
        Pred=tf.truediv(Numer, Denom) ## (N,) ##
        del Numer, Denom0, Denom1, Denom
        return Pred

    def Prediction_Matrix_TF(self, P_User, P_Item, Pr, rating_scales):
        ### Calculating predicted ratings on all items for all users###
        ## P_User: a (num_users, K) array; P(u,Cu) ##
        ## P_Item: a (num_items, L) array; P(v,Cv) ##
        ## Pr    : a (R, K, L) array; P(r|Cu,Cv) ##
        ##rating_scales: a (R, 1) array##
        R=tf.shape(Pr)[0]
        # num_users=tf.shape(P_User)[0]
        # num_items=tf.shape(P_Item)[0]
        Pred=rating_scales[0]*tf.matmul(tf.matmul(P_User, Pr[0]), P_Item, transpose_b=True)
        r=tf.constant(1, dtype=tf.int32)
        def cond(r, Pred):
            return r<R     
        def body(r, Pred):
            Pred=tf.add(Pred, rating_scales[r]*tf.matmul(tf.matmul(P_User, Pr[r]), P_Item, transpose_b=True))
            r+=1
            return [r, Pred]
        _,Pred2=tf.while_loop(cond, body, loop_vars=[r, Pred], \
            shape_invariants=[r.get_shape(), Pred.get_shape()], maximum_iterations=len(self.rating_scales)-1)
        Pu0=tf.reduce_sum(P_User,1)
        Pi0=tf.reduce_sum(P_Item,1)
        Pu=tf.reshape(Pu0+tf.cast(tf.equal(Pu0, 0.0), dtype=tf.float32), shape=[-1,1])
        Pi=tf.reshape(Pi0+tf.cast(tf.equal(Pi0, 0.0), dtype=tf.float32), shape=[1,-1])
        Denom=tf.matmul(Pu,Pi)
        del Pu0, Pu, Pi0, Pi
        Prediction=tf.truediv(Pred2,Denom)
        del Pred2, Denom
        return Prediction
    
    def Target_Function_TF(self, P_User, P_Item, Pr, interval, userind, itemind, ratings, rating_scales):
        split_num=tf.shape(interval)[0]-1
        Error=tf.constant(0.0, dtype=tf.float32)
        ind=tf.constant(0, dtype=tf.int32)
        def cond(ind, Error):
            return ind<split_num
        def body(ind, Error):
            start=tf.gather_nd(interval, [ind,])
            end=tf.gather_nd(interval, [ind+1,])
            num=end-start 
            users=tf.slice(userind, [start,0], [num,1])
            items=tf.slice(itemind, [start,0], [num,1])
            rat=tf.slice(ratings, [start,], [num,])
            Pred=self.Prediction_TF(P_User, P_Item, Pr, users, items, rating_scales)
            Error+=tf.nn.l2_loss(tf.subtract(Pred, rat))
            ind+=1
            del Pred, rat, users, items, start, end, num
            return [ind,Error] 
        _,Error2=tf.while_loop(cond, body, loop_vars=[ind, Error], \
            shape_invariants=[ind.get_shape(), Error.get_shape()], maximum_iterations=split_num)        
        return Error2

    def Training_AEM(self,PCu0,PCv0,Pr0,PUC0,PVC0,rating_scales,tol):
        train1_userind_tf=tf.constant(self.train1_userind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train1_itemind_tf=tf.constant(self.train1_itemind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train1_ratingind_tf=tf.constant(self.train1_ratingind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train1_interval_tf=tf.constant(self.train1_interval, dtype=tf.int64) ## (split_num1,) ## 
        train2_userind_tf=tf.constant(self.train2_userind.reshape(-1,1), dtype=tf.int64) ## (Num2,1) ##
        train2_itemind_tf=tf.constant(self.train2_itemind.reshape(-1,1), dtype=tf.int64) ## (Num2,1) ##
        train2_ratings_tf=tf.constant(self.train2_ratings, dtype=tf.float32) ## (Num2,) ##
        train2_interval_tf=tf.constant(self.train2_interval, dtype=tf.int64) ## (split_num2,) ## 
        b=tf.constant(1.0, dtype=tf.float32)
        b0=tf.constant(1.0, dtype=tf.float32)
        ###  Calculating initial error with (PCu0, PCv0, Pr0, PUC0, PVC0)  ############################
        P_User0=tf.multiply(PUC0, PCu0)
        P_Item0=tf.multiply(PVC0, PCv0)
        Error0=self.Target_Function_TF(P_User0, P_Item0, Pr0, train2_interval_tf, train2_userind_tf, \
            train2_itemind_tf, train2_ratings_tf, rating_scales)
        EE=tf.reshape(Error0, shape=[-1,])
        Error_Array=tf.reshape(tf.concat([EE,EE,EE], axis=0), shape=[1,3])

        def cond(b,PCu0, PCv0, PUC0, PVC0, Pr0, b0, Error0, Error_Array):
            return tf.less(0.5, b)
        
        def body(b,PCu0, PCv0, PUC0, PVC0, Pr0, b0, Error0, Error_Array):
            Error_Array2, PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0=self._BEM(PCu0, PCv0, PUC0, PVC0, Pr0, Error0, \
                train1_interval_tf, train1_userind_tf, train1_itemind_tf, train1_ratingind_tf, \
                    train2_interval_tf, train2_userind_tf, train2_itemind_tf, train2_ratings_tf, rating_scales, b, b0, tol)
            b=b*0.9
            Error_Array=tf.concat([Error_Array,Error_Array2], axis=0)
            return [b,PCu0, PCv0, PUC0, PVC0, Pr0, b0, Error0, Error_Array]
        
        _, PCu_New, PCv_New, PUC_New, PVC_New, Pr_New, b_New, _, Error_Array_New=tf.while_loop(cond, body, \
            loop_vars=[b,PCu0, PCv0, PUC0, PVC0, Pr0, b0, Error0, Error_Array], \
                shape_invariants=[b.get_shape(), tf.TensorShape([None,]), tf.TensorShape([None,]), tf.TensorShape([None, None]), \
                    tf.TensorShape([None, None]), tf.TensorShape([None, None, None]), b0.get_shape(), Error0.get_shape(), \
                        tf.TensorShape([None, 3])],maximum_iterations=7)
        return PCu_New, PCv_New, Pr_New, PUC_New, PVC_New,  b_New, Error_Array_New
    
    def Training_EM(self,PCu0,PCv0,Pr0,PUC0,PVC0,b,rating_scales,tol):
        train_userind_tf=tf.constant(self.train_userind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train_itemind_tf=tf.constant(self.train_itemind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train_ratingind_tf=tf.constant(self.train_ratingind.reshape(-1,1), dtype=tf.int64) ## (Num1,1) ##
        train_interval_tf=tf.constant(self.train_interval, dtype=tf.int64) ## (split_num1,) ## 
        val_userind_tf=tf.constant(self.train_userind.reshape(-1,1), dtype=tf.int64) ## (Num2,1) ##
        val_itemind_tf=tf.constant(self.train_itemind.reshape(-1,1), dtype=tf.int64) ## (Num2,1) ##
        val_ratings_tf=tf.constant(self.train_ratings, dtype=tf.float32) ## (Num2,) ##
        val_interval_tf=tf.constant(self.train_interval, dtype=tf.int64) ## (split_num2,) ## 
        ###  Calculating initial error with (PCu0, PCv0, Pr0, PUC0, PVC0)  ############################
        P_User0=tf.multiply(PUC0, PCu0)
        P_Item0=tf.multiply(PVC0, PCv0)
        Error0=self.Target_Function_TF(P_User0, P_Item0, Pr0, val_interval_tf, val_userind_tf, \
            val_itemind_tf, val_ratings_tf, rating_scales)
        EE=tf.reshape(Error0, shape=[-1,])
        Error_Array=tf.reshape(tf.concat([EE,EE,EE], axis=0), shape=[1,3])
        Error_Array2, PCu_New, PCv_New, Pr_New, PUC_New, PVC_New, _, _=self._BEM(PCu0, PCv0, PUC0, PVC0, Pr0, Error0, \
            train_interval_tf, train_userind_tf, train_itemind_tf, train_ratingind_tf, \
                val_interval_tf, val_userind_tf, val_itemind_tf, val_ratings_tf, rating_scales, b, b, tol) 
        Error_Array_New=tf.concat([Error_Array,Error_Array2], axis=0)
        return PCu_New, PCv_New, Pr_New, PUC_New, PVC_New, Error_Array_New
    

    def _BEM(self, PCu0, PCv0, PUC0, PVC0, Pr0, Error0, train1_interval, train1_userind,\
        train1_itemind, train1_ratingind, train2_interval, train2_userind, \
            train2_itemind, train2_ratings, rating_scales, b, b0, tol):
        RKL=tf.shape(Pr0)
        R=RKL[0]
        K=RKL[1]
        L=RKL[2]
        Nu=tf.shape(PUC0)[0]
        Ni=tf.shape(PVC0)[0]
        PCu_Numer=tf.zeros([K,], dtype=tf.float32)
        PCv_Numer=tf.zeros([L,], dtype=tf.float32)
        PUC_Numer=tf.zeros([Nu, K], dtype=tf.float32)
        PVC_Numer=tf.zeros([Ni, L], dtype=tf.float32)
        Pr_Numer=tf.zeros([R, K*L], dtype=tf.float32)
        P_User0=tf.multiply(PUC0, PCu0)
        P_Item0=tf.multiply(PVC0, PCv0)
        ### Iterating once to produce (PCu1, PCv1, Pr1, PUC1, PVC1) ####
        PCu1,PCv1,PUC1,PVC1,Pr1=self.Expectation_Maximization_TF(\
            PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer, P_User0, P_Item0, Pr0, \
                train1_interval, train1_userind, train1_itemind, train1_ratingind, b)
        P_User1=tf.multiply(PUC1, PCu1)
        P_Item1=tf.multiply(PVC1, PCv1)
        Error1=self.Target_Function_TF(P_User1, P_Item1, Pr1, train2_interval, train2_userind, \
            train2_itemind, train2_ratings, rating_scales)
        diff=Error0-Error1
        Error_Array=tf.reshape(tf.concat([[b], [Error1], [diff]], axis=0), shape=[1,3])

        def cond(diff, Error_Array, PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0, PCu1, PCv1, Pr1, PUC1, PVC1, P_User1, P_Item1, Error1):
            return diff>tol 
        
        def body(diff, Error_Array, PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0, PCu1, PCv1, Pr1, PUC1, PVC1, P_User1, P_Item1, Error1):
            del PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0, diff
            PCu0=PCu1 
            PCv0=PCv1 
            Pr0=Pr1 
            PUC0=PUC1 
            PVC0=PVC1 
            b0=b
            Error0=Error1
            P_User0=P_User1 
            P_Item0=P_Item1
            PCu1,PCv1,PUC1,PVC1,Pr1=self.Expectation_Maximization_TF(\
                PCu_Numer, PCv_Numer, PUC_Numer, PVC_Numer, Pr_Numer, P_User0, P_Item0, Pr0, \
                    train1_interval, train1_userind, train1_itemind, train1_ratingind, b)
            P_User1=tf.multiply(PUC1, PCu1)
            P_Item1=tf.multiply(PVC1, PCv1)
            Error1=self.Target_Function_TF(P_User1, P_Item1, Pr1, train2_interval, train2_userind, \
                train2_itemind, train2_ratings, rating_scales)
            diff=Error0-Error1
            EE=tf.reshape(tf.concat([[b], [Error1], [diff]], axis=0), shape=[1,3])
            Error_Array=tf.concat([Error_Array,EE], axis=0)
            del EE
            return [diff, Error_Array, PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0, PCu1, PCv1, Pr1, PUC1, PVC1, P_User1, P_Item1, Error1]
        _, Error_Array_New, PCu, PCv, Pr, PUC, PVC, Error, bb, _, _, _, _, _, _, _, _=tf.while_loop(cond, body, \
            loop_vars=[diff, Error_Array, PCu0, PCv0, Pr0, PUC0, PVC0, Error0, b0, PCu1, PCv1, Pr1, PUC1, PVC1, P_User1, P_Item1, Error1], \
                shape_invariants=[diff.get_shape(), tf.TensorShape([None, 3]), tf.TensorShape([None,]), tf.TensorShape([None,]), tf.TensorShape([None, None, None]), \
                    tf.TensorShape([None, None]), tf.TensorShape([None, None]), Error0.get_shape(), b0.get_shape(), tf.TensorShape([None,]), tf.TensorShape([None,]), tf.TensorShape([None, None, None]), \
                        tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), Error1.get_shape()],\
                            maximum_iterations=self.max_epoches)        
        return Error_Array_New, PCu, PCv, Pr, PUC, PVC, Error, bb



class MHF_TFCP(object):
    def __init__(self, Train_Matrices, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Matrix, Val_Rank_Mask, \
        U0, V0, D0, P0, Q0, weights, T, lam1, lam2, max_epoches=500, seed=42, std=0.1, tol=1e-2):
        self.Train_Matrices0, self.Train_Masks=Train_Matrices, Train_Masks
        self.Val_Matrix, self.Val_Mask=Val_Matrix,Val_Mask 
        self.Val_Rank_Matrix, self.Val_Rank_Mask=Val_Rank_Matrix, Val_Rank_Mask
        self.val_rank_items=np.array(list(map(lambda x: Val_Rank_Matrix[x].nonzero()[0], range(Val_Matrix.shape[0]))), dtype=object)
        self.U0,self.V0,self.D0=U0,V0,D0 
        self.P0,self.Q0=P0,Q0 
        self.weights=weights 
        self.lam1=lam1 
        self.lam2=lam2
        self.max_epoches, self.seed, self.tol=max_epoches, seed, tol
        self.K,self.R=D0.shape
        self.T=T
        self.num_users_list=[Train_Matrices[k].shape[0] for k in range(self.K)]
        self.num_items_list=[Train_Matrices[k].shape[1] for k in range(self.K)]
        self.bu0=[np.zeros(num_users) for num_users in self.num_users_list]
        self.bv0=[np.zeros(num_items) for num_items in self.num_items_list]
        np.random.seed(seed)
        self.A0=[np.random.normal(0, std, num_users*self.T).reshape(-1,self.T) for num_users in self.num_users_list]
        self.B0=[np.random.normal(0, std, num_items*self.T).reshape(-1,self.T) for num_items in self.num_items_list]
        self.Aver=[Train_Matrices[k].sum()/Train_Masks[k].sum() for k in range(self.K)]
        self.Train_Matrices=[Train_Matrices[k]-self.Aver[k] for k in range(self.K)]
        return 
    
    def reset_graph(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        return
    
    def Updating_QPUVD(self, Train_Matrices, UDV0, P0, U0, V0, D0, E, lam1, weights):
        ## UDV0, D0: with wk and beta ##
        #### Updating Q #####
        QQQ=[tf.linalg.svd(tf.matmul(tf.matmul(P0[k], UDV0[k]), Train_Matrices[k], transpose_a=True), full_matrices=False) \
            for k in range(self.K)]
        Q=[tf.matmul(QQQ[k][2], QQQ[k][1], transpose_b=True) for k in range(self.K)]
        del QQQ, P0
        #### Updatubg P ####
        XQ=[tf.matmul(Train_Matrices[k], Q[k]) for k in range(self.K)]
        PPP=[tf.linalg.svd(tf.matmul(UDV0[k], XQ[k], transpose_b=True), full_matrices=False) \
            for k in range(self.K)]
        P=[tf.matmul(PPP[k][2], PPP[k][1], transpose_b=True) for k in range(self.K)]
        del PPP
        #### Updating U,V,D ####
        Y=tf.stack([tf.matmul(P[k], XQ[k], transpose_a=True) for k in range(self.K)]) ## (K, R, R) ###
        del XQ
        UUU=tf.linalg.inv(tf.multiply(tf.matmul(D0,D0,transpose_a=True), tf.matmul(V0,V0,transpose_a=True))+lam1*E)
        U=tf.matmul(tf.matmul(Basic_Operations.Tensor2Matrix1_GPU(Y), Basic_Operations.Khatri_Rao_GPU(D0,V0)), UUU)
        del UUU, U0
        VVV=tf.linalg.inv(tf.multiply(tf.matmul(D0,D0,transpose_a=True), tf.matmul(U,U,transpose_a=True))+lam1*E)
        V=tf.matmul(tf.matmul(Basic_Operations.Tensor2Matrix2_GPU(Y), Basic_Operations.Khatri_Rao_GPU(D0,U)), VVV)
        del VVV, V0
        Y3=Basic_Operations.Tensor2Matrix3_GPU(Y) 
        del Y, D0
        VU=Basic_Operations.Khatri_Rao_GPU(V,U)
        VVUU=tf.multiply(tf.matmul(V,V,transpose_a=True), tf.matmul(U,U,transpose_a=True))
        D=tf.concat([tf.matmul(tf.matmul(tf.reshape(Y3[k], shape=[1,-1]), VU), \
            tf.linalg.inv(weights[k]*VVUU+weights[k]*lam1*E)) for k in range(self.K)], axis=0) ##without weights##
        del Y3, VU, VVUU, UDV0
        return Q,P,U,V,D
    
    def Updating_Bias(self, Error_Mat, Buw0, Bvw0, lam, weights):
        ## Buw0, Bvw0: with wk ##
        Bu_Denom=[1/(tf.cast(Bvw0[k].shape[0], dtype=tf.float32)*weights[k]+weights[k]*lam) for k in range(self.K)]
        BBv=[tf.tile([tf.reduce_sum(Bvw0[k]),], Buw0[k].shape) for k in range(self.K)]
        Bu=[Bu_Denom[k]*(tf.reduce_sum(Error_Mat[k], 1)-BBv[k]) for k in range(self.K)] ##without weights##
        del Bu_Denom, BBv

        Bv_Denom=[1/(tf.cast(Buw0[k].shape[0], dtype=tf.float32)*weights[k]+weights[k]*lam) for k in range(self.K)]
        BBu=[tf.tile([weights[k]*tf.reduce_sum(Bu[k]),], Bvw0[k].shape) for k in range(self.K)]
        Bv=[Bv_Denom[k]*(tf.reduce_sum(Error_Mat[k], 0)-BBu[k]) for k in range(self.K)] ##without weights##
        del Bv_Denom, BBu
        return Bu,Bv
    
    def Updating_AkBk(self, Error_Mat, A0, B0, lam, weights):
        ##  ##
        E=tf.eye(self.T)
        Adenom=[tf.matmul(B0[k], B0[k], transpose_a=True)*weights[k]+weights[k]*lam*E \
            for k in range(self.K)]
        A=[tf.matmul(tf.matmul(Error_Mat[k], B0[k]), tf.linalg.inv(Adenom[k])) \
            for k in range(self.K)]
        Bdenom=[tf.matmul(A[k], A[k], transpose_a=True)*weights[k]+weights[k]*lam*E \
            for k in range(self.K)]
        B=[tf.matmul(tf.matmul(tf.transpose(Error_Mat[k]), A[k]), tf.linalg.inv(Bdenom[k])) \
            for k in range(self.K)]
        del E, Adenom, Bdenom
        return A, B

    def First_Iteration(self, Train_Matrices_Init, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Mask, \
        Buw0, Bvw0, A0, B0, P0, Q0, U0, V0, Dw0, E, Aver, lam1, lam2, weights, Rating_Pow, Denom_Log, val_idcg, val_userd_float):
        UDV0=[tf.matmul(tf.matmul(U0, tf.linalg.diag(Dw0[k])), V0, transpose_b=True) for k in range(self.K)] ## with wk and beta##
        Bias_User0=[tf.tile(tf.reshape(Buw0[k],shape=[-1,1]),[1,Bvw0[k].shape[0]]) for k in range(self.K)] ## with wk ##
        Bias_Item0=[tf.tile(tf.reshape(Bvw0[k],shape=[1,-1]),[Buw0[k].shape[0],1]) for k in range(self.K)] ## with wk ##
        AB0=[weights[k]*tf.matmul(A0[k], B0[k], transpose_b=True) for k in range(self.K)] ## with wk ##
        del Q0
        Z1=[Train_Matrices_Init[k]-Bias_User0[k]-Bias_Item0[k]-AB0[k] for k in range(self.K)]
        Q,P,U,V,D=self.Updating_QPUVD(Z1, UDV0, P0, U0, V0, Dw0, E, lam1, weights) ##OutPut D without weights##
        del Z1, AB0
        UDV=[tf.matmul(tf.matmul(U, weights[k]*tf.linalg.diag(D[k])), V, transpose_b=True) for k in range(self.K)] 
        PUDVQ=[tf.matmul(tf.matmul(P[k], UDV[k]), Q[k], transpose_b=True) for k in range(self.K)] ## with weights ##
        ####  Update Ak and Bk  ##################
        Z2=[Train_Matrices_Init[k]-PUDVQ[k]-Bias_User0[k]-Bias_Item0[k] for k in range(self.K)]
        A,B=self.Updating_AkBk(Z2, A0, B0, lam2, weights) ##A,B: without wk##
        del Z2, A0, B0
        AB=[weights[k]*tf.matmul(A[k], B[k], transpose_b=True) for k in range(self.K)] ## with weights ##
        ####  Update Bu and Bv  ########
        Z4=[Train_Matrices_Init[k]-PUDVQ[k]-AB[k] for k in range(self.K)]
        Bu,Bv=self.Updating_Bias(Z4, Buw0, Bvw0, lam2, weights) ## Bu, Bv: without wk ##
        del Z4, Buw0, Bvw0, Bias_User0, Bias_Item0
        Bias_User=[tf.tile(tf.reshape(weights[k]*Bu[k],shape=[-1,1]),[1,Bv[k].shape[0]]) for k in range(self.K)]
        Bias_Item=[tf.tile(tf.reshape(weights[k]*Bv[k],shape=[1,-1]),[Bu[k].shape[0],1]) for k in range(self.K)]
        ###### Calcluating loss #####################
        Pred=[PUDVQ[k]+AB[k]+Bias_User[k]+Bias_Item[k] for k in range(self.K)] ## with weights ##
        del PUDVQ, AB, Bias_User, Bias_Item
        Pred2=tf.truediv(Pred[-1]+Aver[-1], weights[-1])
        Loss0=tf.reduce_sum(tf.stack([tf.nn.l2_loss(Train_Matrices_Init[k]-Pred[k]) for k in range(self.K)]))
        Loss1=tf.reduce_sum(tf.stack([tf.nn.l2_loss(tf.multiply(Train_Masks[k], Train_Matrices_Init[k]-Pred[k])) \
            for k in range(self.K)]))
        del Pred
        Loss2=tf.nn.l2_loss(U)+tf.nn.l2_loss(V)
        Loss3=tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*D[k]) for k in range(self.K)]))
        Loss4=tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*Bu[k]) for k in range(self.K)]))\
            +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*Bv[k]) for k in range(self.K)]))\
                +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*A[k]) for k in range(self.K)]))\
                    +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*B[k]) for k in range(self.K)]))
        LossA=Loss0+lam1*Loss2+lam1*Loss3+lam2*Loss4
        LossB=Loss1+lam1*Loss2+lam1*Loss3+lam2*Loss4
        RMSE=tf.sqrt(tf.truediv(tf.reduce_sum(tf.squared_difference(Val_Matrix, tf.multiply(Pred2, Val_Mask))), tf.reduce_sum(Val_Mask)))
        NDCG,Pre,Re,NDCGN=Basic_Operations.Validation_Rank_GPU(\
            Pred2, Train_Masks[-1], Val_Rank_Mask, Rating_Pow, Denom_Log, val_idcg, val_userd_float)
        Loss_Array=tf.reshape(tf.concat([[LossA,], [LossB,], [Loss2,], [Loss3,], [Loss4,], [RMSE,], [NDCG,], [Pre,], [Re,], [NDCGN,]], axis=0), shape=[1,-1])
        del Loss0, Loss1, Loss2, Loss3, Loss4, LossA, LossB, Pred2, RMSE, NDCG, Pre, Re, NDCGN
        return Bu,Bv,A,B,Q,P,U,V,D,Loss_Array
    
    
    def Following_Iteration(self, Train_Matrices_Init, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Mask, \
        Buw0, Bvw0, A0, B0, P0, Q0, U0, V0, Dw0, E, Aver, lam1, lam2, weights, Rating_Pow, Denom_Log, val_idcg, val_userd_float):
        UDV0=[tf.matmul(tf.matmul(U0, tf.linalg.diag(Dw0[k])), V0, transpose_b=True) for k in range(self.K)] ## with wk and beta ##
        ###Complement missing values###
        Bias_User0=[tf.tile(tf.reshape(Buw0[k],shape=[-1,1]),[1,Bvw0[k].shape[0]]) for k in range(self.K)] ## with wk ##
        Bias_Item0=[tf.tile(tf.reshape(Bvw0[k],shape=[1,-1]),[Buw0[k].shape[0],1]) for k in range(self.K)] ## with wk ##
        AB0=[weights[k]*tf.matmul(A0[k], B0[k], transpose_b=True) for k in range(self.K)] ## with wk ##
        Pred0=[tf.matmul(tf.matmul(P0[k], UDV0[k]), Q0[k], transpose_b=True)+AB0[k]+Bias_User0[k]+Bias_Item0[k] for k in range(self.K)] ## with weights ##
        Train_Matrices=[tf.multiply(Train_Matrices_Init[k], Train_Masks[k])+tf.multiply(Pred0[k], 1.0-Train_Masks[k]) for k in range(self.K)]
        del Pred0, Q0
        Z1=[Train_Matrices[k]-Bias_User0[k]-Bias_Item0[k]-AB0[k] for k in range(self.K)]
        Q,P,U,V,D=self.Updating_QPUVD(Z1, UDV0, P0, U0, V0, Dw0, E, lam1, weights) ##OutPut D without weights##
        del Z1, AB0
        UDV=[tf.matmul(tf.matmul(U, weights[k]*tf.linalg.diag(D[k])), V, transpose_b=True) for k in range(self.K)]
        PUDVQ=[tf.matmul(tf.matmul(P[k], UDV[k]), Q[k], transpose_b=True) for k in range(self.K)] ## with weights ##
        ####  Update Ak and Bk  ##################
        Z2=[Train_Matrices[k]-PUDVQ[k]-Bias_User0[k]-Bias_Item0[k] for k in range(self.K)]
        A,B=self.Updating_AkBk(Z2, A0, B0, lam2, weights) ##A,B: without wk##
        del Z2, A0, B0
        AB=[weights[k]*tf.matmul(A[k], B[k], transpose_b=True) for k in range(self.K)] ## with weights ##
        ####  Update Bu and Bv  ########
        Z4=[Train_Matrices[k]-PUDVQ[k]-AB[k] for k in range(self.K)]
        Bu,Bv=self.Updating_Bias(Z4, Buw0, Bvw0, lam2, weights) ## Bu, Bv: without wk ##
        del Z4, Buw0, Bvw0, Bias_User0, Bias_Item0
        Bias_User=[tf.tile(tf.reshape(weights[k]*Bu[k],shape=[-1,1]),[1,Bv[k].shape[0]]) for k in range(self.K)]
        Bias_Item=[tf.tile(tf.reshape(weights[k]*Bv[k],shape=[1,-1]),[Bu[k].shape[0],1]) for k in range(self.K)]
        ###### Calcluating loss #####################
        Pred=[PUDVQ[k]+AB[k]+Bias_User[k]+Bias_Item[k] for k in range(self.K)] ## with weights ##
        del PUDVQ, AB, Bias_User, Bias_Item
        Pred2=tf.truediv(Pred[-1]+Aver[-1], weights[-1])
        Loss0=tf.reduce_sum(tf.stack([tf.nn.l2_loss(Train_Matrices[k]-Pred[k]) for k in range(self.K)]))
        Loss1=tf.reduce_sum(tf.stack([tf.nn.l2_loss(tf.multiply(Train_Masks[k], Train_Matrices[k]-Pred[k])) \
            for k in range(self.K)]))
        del Pred, Train_Matrices
        Loss2=tf.nn.l2_loss(U)+tf.nn.l2_loss(V)
        Loss3=tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*D[k]) for k in range(self.K)]))
        Loss4=tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*Bu[k]) for k in range(self.K)]))\
            +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*Bv[k]) for k in range(self.K)]))\
                +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*A[k]) for k in range(self.K)]))\
                    +tf.reduce_sum(tf.stack([tf.nn.l2_loss(weights[k]*B[k]) for k in range(self.K)]))
        LossA=Loss0+lam1*Loss2+lam1*Loss3+lam2*Loss4
        LossB=Loss1+lam1*Loss2+lam1*Loss3+lam2*Loss4
        RMSE=tf.sqrt(tf.truediv(tf.reduce_sum(tf.squared_difference(Val_Matrix, tf.multiply(Pred2, Val_Mask))), tf.reduce_sum(Val_Mask)))
        NDCG,Pre,Re,NDCGN=Basic_Operations.Validation_Rank_GPU(\
            Pred2, Train_Masks[-1], Val_Rank_Mask, Rating_Pow, Denom_Log, val_idcg, val_userd_float)
        Loss_Array=tf.reshape(tf.concat([[LossA,], [LossB,], [Loss2,], [Loss3,], [Loss4,], [RMSE,], [NDCG,], [Pre,], [Re,], [NDCGN,]], axis=0), shape=[1,-1])
        del Loss0, Loss1, Loss2, Loss3, Loss4, LossA, LossB, Pred2, RMSE, NDCG, Pre, Re, NDCGN
        return Bu,Bv,A,B,Q,P,U,V,D,Loss_Array
    
    def Transfer_Learning(self):
        self.reset_graph()
        Val_Matrix=tf.constant(self.Val_Matrix, dtype=tf.float32)
        Val_Mask=tf.constant(self.Val_Mask, dtype=tf.float32)
        Val_Rank_Matrix=tf.constant(self.Val_Rank_Matrix, dtype=tf.float32)
        Val_Rank_Mask=tf.constant(self.Val_Rank_Mask, dtype=tf.float32)
        val_userd_float0=tf.reduce_sum(Val_Rank_Mask, 1)
        val_userd_float=val_userd_float0+tf.cast(tf.equal(val_userd_float0, 0.0), dtype=tf.float32)
        val_idcg0=tf.constant(Basic_Operations.Calculating_IDCG(self.Val_Rank_Matrix, \
            self.Val_Rank_Mask, self.val_rank_items), dtype=tf.float32)
        val_idcg=val_idcg0+tf.cast(tf.equal(val_idcg0, 0.0), dtype=tf.float32)
        Rating_Pow=tf.pow(2.0, Val_Rank_Matrix)-1.0 ## (num_users, num_items) ##
        Denom_Log=tf.constant(np.array([np.log2(x+2) for x in range(20)], dtype=object), dtype=tf.float32)
        Train_Matrices_Init=[tf.constant(self.weights[k]*self.Train_Matrices[k], dtype=tf.float32) \
            for k in range(self.K)] ## with weights ##
        Train_Masks=[tf.constant(Mat, dtype=tf.float32) for Mat in self.Train_Masks]
        weights=tf.constant(self.weights, dtype=tf.float32)
        lam1=tf.constant(self.lam1, dtype=tf.float32)
        lam2=tf.constant(self.lam2, dtype=tf.float32)
        E=tf.eye(self.R)
        Aver=[tf.constant(self.weights[k]*self.Aver[k], dtype=tf.float32) for k in range(self.K)]
        BBu0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_users_list),]) ## with wk ##
        BBv0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_items_list),]) ## with wk ##
        U0=tf.placeholder(dtype=tf.float32, shape=[self.R, self.R])
        V0=tf.placeholder(dtype=tf.float32, shape=[self.R, self.R])
        Dw0=tf.placeholder(dtype=tf.float32, shape=[self.K, self.R]) ## with wk and beta ##
        PP0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_users_list), self.R])
        QQ0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_items_list), self.R])
        AA0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_users_list), self.T])
        BB0=tf.placeholder(dtype=tf.float32, shape=[sum(self.num_items_list), self.T])
        Buw0=[tf.slice(BBu0, [sum(self.num_users_list[:k]),], [self.num_users_list[k],]) for k in range(self.K)] ## with wk ##
        Bvw0=[tf.slice(BBv0, [sum(self.num_items_list[:k]),], [self.num_items_list[k],]) for k in range(self.K)] ## with wk ##
        P0=[tf.slice(PP0, [sum(self.num_users_list[:k]), 0], [self.num_users_list[k], self.R]) \
            for k in range(self.K)]
        Q0=[tf.slice(QQ0, [sum(self.num_items_list[:k]), 0], [self.num_items_list[k], self.R]) \
            for k in range(self.K)]
        A0=[tf.slice(AA0, [sum(self.num_users_list[:k]), 0], [self.num_users_list[k], self.T])\
            for k in range(self.K)] ## without weights##
        B0=[tf.slice(BB0, [sum(self.num_items_list[:k]), 0], [self.num_items_list[k], self.T]) \
            for k in range(self.K)] ## without weights ##
        Bu1,Bv1,A1,B1,Q1,P1,U1,V1,D1,Loss_Array1=self.First_Iteration(Train_Matrices_Init, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Mask, \
            Buw0, Bvw0, A0, B0, P0, Q0, U0, V0, Dw0, E, Aver, lam1, lam2, weights, Rating_Pow, Denom_Log, val_idcg, val_userd_float)
        Bu2,Bv2,A2,B2,Q2,P2,U2,V2,D2,Loss_Array2=self.Following_Iteration(Train_Matrices_Init, Train_Masks, Val_Matrix, Val_Mask, Val_Rank_Mask, \
            Buw0, Bvw0, A0, B0, P0, Q0, U0, V0, Dw0, E, Aver, lam1, lam2, weights, Rating_Pow, Denom_Log, val_idcg, val_userd_float)
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            loss_array=np.zeros((self.max_epoches, 10))
            bbu0=np.hstack([self.weights[k]*self.bu0[k] for k in range(self.K)])
            bbv0=np.hstack([self.weights[k]*self.bv0[k] for k in range(self.K)])
            aa0=np.vstack(self.A0)
            bb0=np.vstack(self.B0)
            u0, v0=self.U0, self.V0
            d0=np.multiply(self.D0.T, self.weights).T 
            pp0=np.vstack(self.P0)
            qq0=np.vstack(self.Q0)
            bu1,bv1,a1,b1,q1,p1,u1,v1,d1,loss_array[0]=sess.run([Bu1,Bv1,A1,B1,Q1,P1,U1,V1,D1,Loss_Array1], \
                feed_dict={BBu0:bbu0, BBv0:bbv0, AA0: aa0, BB0: bb0, U0:u0, V0:v0, Dw0:d0, PP0:pp0, QQ0:qq0})
            minrmse_itertime=0
            minrmse_bu,minrmse_bv=bu1,bv1
            minrmse_a,minrmse_b=a1,b1
            minrmse_u,minrmse_v,minrmse_d=u1,v1,d1 
            minrmse_p,minrmse_q=p1,q1
            maxndcg_itertime=0
            maxndcg_bu,maxndcg_bv=bu1,bv1
            maxndcg_a,maxndcg_b=a1,b1
            maxndcg_u,maxndcg_v,maxndcg_d=u1,v1,d1 
            maxndcg_p,maxndcg_q=p1,q1
            maxre20_itertime=0
            maxre20_bu,maxre20_bv=bu1,bv1
            maxre20_a,maxre20_b=a1,b1
            maxre20_u,maxre20_v,maxre20_d=u1,v1,d1 
            maxre20_p,maxre20_q=p1,q1
            maxndcg20_itertime=0
            maxndcg20_bu,maxndcg20_bv=bu1,bv1
            maxndcg20_a,maxndcg20_b=a1,b1
            maxndcg20_u,maxndcg20_v,maxndcg20_d=u1,v1,d1 
            maxndcg20_p,maxndcg20_q=p1,q1
            for epoch in range(1, self.max_epoches):
                bu2,bv2,a2,b2,q2,p2,u2,v2,d2,loss_array[epoch]=sess.run([Bu2,Bv2,A2,B2,Q2,P2,U2,V2,D2,Loss_Array2], \
                    feed_dict={BBu0:np.hstack([self.weights[k]*bu1[k] for k in range(self.K)]), \
                        BBv0:np.hstack([self.weights[k]*bv1[k] for k in range(self.K)]), \
                            AA0:np.vstack(a1), BB0:np.vstack(b1), U0:u1, V0:v1, Dw0:np.multiply(d1.T, self.weights).T, \
                                PP0:np.vstack(p1), QQ0:np.vstack(q1)})
                if loss_array[epoch,-5]<loss_array[minrmse_itertime,-5]:
                    minrmse_itertime=epoch
                    minrmse_bu,minrmse_bv=bu2,bv2
                    minrmse_a,minrmse_b=a2,b2
                    minrmse_u,minrmse_v,minrmse_d=u2,v2,d2 
                    minrmse_p,minrmse_q=p2,q2
                if loss_array[epoch,-4]>loss_array[maxndcg_itertime,-4]:
                    maxndcg_itertime=epoch
                    maxndcg_bu,maxndcg_bv=bu2,bv2
                    maxndcg_a,maxndcg_b=a2,b2
                    maxndcg_u,maxndcg_v,maxndcg_d=u2,v2,d2  
                    maxndcg_p,maxndcg_q=p2,q2
                if loss_array[epoch,-2]>loss_array[maxre20_itertime,-2]:
                    maxre20_itertime=epoch
                    maxre20_bu,maxre20_bv=bu2,bv2
                    maxre20_a,maxre20_b=a2,b2
                    maxre20_u,maxre20_v,maxre20_d=u2,v2,d2  
                    maxre20_p,maxre20_q=p2,q2
                if loss_array[epoch,-1]>loss_array[maxndcg20_itertime,-1]:
                    maxndcg20_itertime=epoch
                    maxndcg20_bu,maxndcg20_bv=bu2,bv2
                    maxndcg20_a,maxndcg20_b=a2,b2
                    maxndcg20_u,maxndcg20_v,maxndcg20_d=u2,v2,d2  
                    maxndcg20_p,maxndcg20_q=p2,q2
                if loss_array[epoch-1,0]-loss_array[epoch,0]<=self.tol and epoch>1:
                    break
                diff_epoch=epoch-np.max([maxndcg_itertime, maxre20_itertime, maxndcg20_itertime])
                if diff_epoch>=100:
                    break
                bu1,bv1,a1,b1,q1,p1,u1,v1,d1=bu2,bv2,a2,b2,q2,p2,u2,v2,d2
        self.Loss_Array=loss_array[:(epoch+1)] 
        self.bu, self.bv=bu2,bv2
        self.A, self.B=a2,b2
        self.P, self.Q=p2,q2 
        self.U, self.D, self.V=u2,d2,v2 
        self.MinRMSE_IterTime=minrmse_itertime
        self.MinRMSE_bu, self.MinRMSE_bv=minrmse_bu, minrmse_bv
        self.MinRMSE_A, self.MinRMSE_B=minrmse_a,minrmse_b
        self.MinRMSE_P, self.MinRMSE_Q=minrmse_p, minrmse_q
        self.MinRMSE_U, self.MinRMSE_D, self.MinRMSE_V=minrmse_u, minrmse_d, minrmse_v
        self.MaxNDCG_IterTime=maxndcg_itertime
        self.MaxNDCG_bu, self.MaxNDCG_bv=maxndcg_bu, maxndcg_bv
        self.MaxNDCG_A, self.MaxNDCG_B=maxndcg_a,maxndcg_b
        self.MaxNDCG_P, self.MaxNDCG_Q=maxndcg_p, maxndcg_q
        self.MaxNDCG_U, self.MaxNDCG_D, self.MaxNDCG_V=maxndcg_u, maxndcg_d, maxndcg_v
        self.MaxRe20_IterTime=maxre20_itertime
        self.MaxRe20_bu, self.MaxRe20_bv=maxre20_bu,maxre20_bv
        self.MaxRe20_A, self.MaxRe20_B=maxre20_a,maxre20_b
        self.MaxRe20_P, self.MaxRe20_Q=maxre20_p, maxre20_q
        self.MaxRe20_U, self.MaxRe20_D, self.MaxRe20_V=maxre20_u, maxre20_d, maxre20_v
        self.MaxNDCG20_IterTime=maxndcg20_itertime
        self.MaxNDCG20_bu,self.MaxNDCG20_bv=maxndcg20_bu, maxndcg20_bv
        self.MaxNDCG20_A, self.MaxNDCG20_B=maxndcg20_a,maxndcg20_b
        self.MaxNDCG20_P, self.MaxNDCG20_Q=maxndcg20_p, maxndcg20_q
        self.MaxNDCG20_U, self.MaxNDCG20_D, self.MaxNDCG20_V=maxndcg20_u, maxndcg20_d, maxndcg20_v
        return 
