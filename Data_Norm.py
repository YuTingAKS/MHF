import numpy as np 
import scipy.sparse

DataName='MovieLens20M'
OutPut=np.load('.\Data\Filtering_{0}_Results.npy'.format(DataName,), allow_pickle=True).tolist()
Matrix=OutPut['After Filtering: Matrix']
indx,indy=Matrix.nonzero()
ratings=np.ceil(np.array(Matrix[indx,indy])[0])
Matrix2=scipy.sparse.csr_matrix((ratings, (indx,indy)), shape=Matrix.shape)
OutPut['After Filtering: Matrix']=Matrix2
np.save('.\Data\Filtering_{0}_Results.npy'.format('ML20M',), OutPut)

