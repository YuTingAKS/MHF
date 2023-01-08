# MHF

# Title：A mixed heterogeneous factorization model for non-overlapping cross-domain recommendation
- Citation：Yu T, Guo J, Li W, et al. A mixed heterogeneous factorization model for non-overlapping cross-domain recommendation[J]. Decision Support Systems, 2021, 151: 113625.
- **Download example data from** https://pan.baidu.com/s/1WSSTKD-3m8LmN9Zm73UM4A?pwd=1ac1 

## Experimental Scenario Setting: 
- Auxiliary domains: ML20M，Netflix
- Target domain: AmazonBooks
- Sparsity: Given100%

## Details for each file:
***
### Data：
Raw data:
- MovieLens20M_Rating_Array.npy
- MovieLens20M_Rating_Matrix.npy
- Netflix_Rating_Array.npy
- Netflix_Rating_Matrix.npy
- AmazonBooks_Rating_Array.npy
- AmazonBooks_Rating_Matrix.npy

Preprocessed data: (**DataSet_Split.py, Data_Norm.py**)
- Filtering_ML20M_Results.npy
- Filtering_Netflix_Results.npy
- Filtering_AmazonBooks_Results.npy

Split data for training, validating and testing: (**Data_Extraction.py**)
- ML20M_Train_Test_Data.npy
- Netflix_Train_Test_Data.npy
- AmazonBooks_Train_Val_Test_Data0.npy

IDCG calculated in advance for NDCG: (**idcg.py**)
- AmazonBooks_idcg0.npy

***
### Subspace_Alignment： 
Initializing Flexible Mixture Model (FMM) by Orthogonal　Nonnegative　Matrix　Tri-Factorization (ONMTF): (**FMMInitial.py**)
- FMMInitial_ML20M_40_40_OutPut.npy
- FMMInitial_Netflix_40_40_OutPut.npy
- FMMInitial_AmazonBooks_Given1.00_40_40_OutPut0.npy

Calculating latent factor matrices for each domain by FMM: (**FMM.py**)
- FMM_ML20M_40_40_OutPut.npy
- FMM_Netflix_40_40_OutPut.npy
- FMM_AmazonBooks_Given1.00_40_40_OutPut0.npy

Inferring information consistency by the subspace alignment technique: (**SAKL.py**)
- SA_MNA_Given1.00_R40_OutPut0.npy

***
### Initialization：

Initializing the group-level latent factor matrices U and V: (**Initialization_Normal_UV.py**)
- Initialization_Normal_UV.npy

Initializing the membership matrices Pk and Qk: (**Initialization_SVD_PQ.py**)
- Initialization_SVD_PQ_ML20M.npy
- Initialization_SVD_PQ_Netflix.npy
- Initialization_SVD_PQ_AmazonBooks_Given1.00_OutPut0.npy


***
### MHF：
Training MHF: (**MHF_Train.py**)
- MHF_MNA_Given1.00_R40_T40_alpha1.0_lam10_lam10_Epoches300_OutPut0.npy
- Note: Adopting early stopping strategy to record the optimal models corresponding to different metrics (RMSE, NDCG, Recall@20, NDCG@20)

Evaluating MHF: (**MHF_Eval.py**)
- MHF_MNA_Given1.00_R40_T40_alpha1.0_lam10_lam10_Eval_OutPut0.npy
- Note: Evaluating all models corresponding to different metrics

***
### Settings: 
Python version: 3.6.13

Dependent packages: **requirement.txt**
