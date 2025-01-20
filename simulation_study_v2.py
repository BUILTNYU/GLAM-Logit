# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:32:14 2022

@author: Xiyuan Ren
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import pickle
import gurobipy as gp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import time
from functions import group_level_IO, one_iteration, the_whole_model_KMeans, gen_unimodal_gaussian, gen_unimodal_gaussian_


#####################################
# Synthetic Data -- Only estimation #
#####################################
def data_generation(a,T,seed=None):
    # Generate dataset
    if a == 'Unimodal':
        mean = [-0.5,-0.5,0.5]
        X,Y,beta_true = gen_unimodal_gaussian(mean,T,seed)
        return X,Y,beta_true
    elif a == 'Multimodal':
        mean1 = [2, 2, 3]
        mean2 = [-0.5,-0.5,0.5]
        mean3 = [-3,-3,-2]
        X1,Y1,beta_true1 = gen_unimodal_gaussian(mean1,int(T/3),seed)
        X2,Y2,beta_true2 = gen_unimodal_gaussian(mean2,int(T/3),seed) 
        X3,Y3,beta_true3 = gen_unimodal_gaussian(mean3,int(T/3),seed)
        X = np.concatenate((X1,X2,X3),axis=0)
        Y = np.concatenate((Y1,Y2,Y3),axis=0)
        beta_true = np.concatenate((beta_true1,beta_true2,beta_true3),axis=0)
        return X,Y,beta_true
    else:
        print('No such setting')

def one_trial_estimation(a,T,k,ini,tol,fix=False):
    # Generate data
    if fix>0:
        X,Y,beta_true = data_generation(a,T,seed=fix)
    else:
        X,Y,beta_true = data_generation(a,T,seed=None)
    # Model estimation
    l_boundary = [-20,-20,-20]
    u_boundary = [20,20,20]
    beta_name = ['a','b','c']
    start_time = time.time()
    beta_array,P_array,cluster_id,cluster_beta_0,iter_num = the_whole_model_KMeans(Y,X,beta_name,ini,l_boundary,u_boundary,tol,k,a)
    end_time = time.time()
    # Calculate Metrics
    computing_time = end_time-start_time
    mean_a = beta_true.mean(axis=0)
    mean_b = beta_array.mean(axis=0)
    beta_RMSE = np.sqrt(np.square(mean_a-mean_b).mean())
    cov_a = np.cov(beta_true.T, bias=False)
    cov_b = np.cov(beta_array.T, bias=False)
    cov_RMSE = np.sqrt(np.square(cov_a-cov_b).mean())
    return beta_RMSE,cov_RMSE,iter_num,computing_time

# # Test
# seed = 8681
# print(one_trial_estimation('Multimodal',500,1,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Multimodal',500,2,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Multimodal',500,3,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Multimodal',500,4,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Multimodal',500,5,[-0.5,-0.5,0.5],0.1,fix=seed))
# print('-------')
# print(one_trial_estimation('Unimodal',500,1,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Unimodal',500,2,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Unimodal',500,3,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Unimodal',500,4,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Unimodal',500,5,[-0.5,-0.5,0.5],0.1,fix=seed))
# print('-------')
# print(one_trial_estimation('Unimodal',500,1,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Unimodal',500,1,[-5,-5,5],0.1,fix=seed))
# print('-------')
# print(one_trial_estimation('Multimodal',500,1,[-0.5,-0.5,0.5],0.1,fix=seed))
# print(one_trial_estimation('Multimodal',500,1,[-5,-5,5],0.1,fix=seed))



# # To plot distribution
a = 'Unimodal'
T = 5000
tol = 0.1
k = 1
ini = [-0.5,-0.5,0.5]
X,Y,beta_true = data_generation(a,T,seed=4825)
1-Y.max(axis=1).sum()/Y.shape[0]

l_boundary = [-20,-20,-20]
u_boundary = [20,20,20]
beta_name = ['a','b','c']
start_time = time.time()
beta_array,P_array,cluster_id,cluster_beta_0,iter_num = the_whole_model_KMeans(Y,X,beta_name,ini,l_boundary,u_boundary,tol,k,a)
end_time = time.time()
# Calculate Metrics
computing_time = end_time-start_time
if k > 1:
    beta_RMSE = 0
    cov_RMSE = 0
    for cluster in range(k):
        mean_a = beta_true[cluster_id==cluster].mean(axis=0)
        mean_b = beta_array[cluster_id==cluster].mean(axis=0)
        beta_RMSE += np.sqrt(np.square(mean_a-mean_b).mean())
        cov_a = np.cov(beta_true[cluster_id==cluster].T, bias=False)
        cov_b = np.cov(beta_array[cluster_id==cluster].T, bias=False)
        cov_RMSE += np.sqrt(np.square(cov_a-cov_b).mean())
    beta_RMSE /= k
    cov_RMSE /= k

fig, ax = plt.subplots(figsize=(8,5))
pd.Series(beta_true[:,2]).hist(bins=60,alpha=0.8,label='True paramters',ax=ax)
pd.Series(beta_array[:,2]).hist(bins=60,alpha=0.8, label='Estimated parameters',ax=ax)
plt.grid(linestyle='--',alpha=0.5)
plt.xlim([-6,6])
plt.ylim([0,350])
plt.legend()
plt.savefig("/Users/ryan/Documents/01.Research Projects/5.Social Equity Project with Replica/11. AMXL paper/3.Transportation Research Part B Revision/Figures/Materials/Fig.1(3).jpg", dpi=300)



# For tables in the paper
beta_RMSE_,cov_RMSE_,iter_num_,computing_time_ = [],[],[],[]
np.random.seed(8521)
random_integers = np.random.randint(1, 10001, size=20)

alpha = 'Unimodal'
T = 500
k = 1
ini = [-0.5,-0.5,0.5]
tol = 0.1

for i in range(20):
    beta_RMSE,cov_RMSE,iter_num,computing_time = one_trial_estimation(alpha,T,k,ini,tol,fix=random_integers[i])
    beta_RMSE_.append(beta_RMSE)
    cov_RMSE_.append(cov_RMSE)
    iter_num_.append(iter_num)
    computing_time_.append(computing_time)
    
beta_RMSE_ = np.array(beta_RMSE_)
cov_RMSE_ = np.array(cov_RMSE_)
iter_num_ = np.array(iter_num_)
computing_time_ = np.array(computing_time_)

print(beta_RMSE_.mean(),beta_RMSE_.std())
print('-------------')
print(cov_RMSE_.mean(),cov_RMSE_.std())
print('-------------')
print(iter_num_.mean(),iter_num_.std())
print('-------------')
print(computing_time_.mean(),computing_time_.std())





#####################################
# Synthetic Data -- With prediction #
#####################################
from sklearn.neighbors import KNeighborsRegressor

def data_generation_(a,T,k,seed=None):
    # Generate dataset
    if a == 'Unimodal':
        mean = [-0.5, -0.5, 0.5, 0,0]
        X,Y,beta_true,xy,xy_test,X_test,Y_test = gen_unimodal_gaussian_(mean,T,k,seed)
        return X,Y,beta_true,xy,xy_test,X_test,Y_test
    elif a == 'Multimodal':
        mean1 = [2, 2, 3,0,0]
        mean2 = [-0.5, -0.5, 0.5,0,0]
        mean3 = [-3,-3,-2,0,0]
        X1,Y1,beta_true1,xy1,xy_test1,X_test1,Y_test1 = gen_unimodal_gaussian_(mean1,int(T/3),k,seed)
        X2,Y2,beta_true2,xy2,xy_test2,X_test2,Y_test2 = gen_unimodal_gaussian_(mean2,int(T/3),k,seed) 
        X3,Y3,beta_true3,xy3,xy_test3,X_test3,Y_test3 = gen_unimodal_gaussian_(mean3,int(T/3),k,seed)
        X = np.concatenate((X1,X2,X3),axis=0)
        Y = np.concatenate((Y1,Y2,Y3),axis=0)
        beta_true = np.concatenate((beta_true1,beta_true2,beta_true3),axis=0)
        xy = np.concatenate((xy1,xy2,xy3),axis=0)
        xy_test = np.concatenate((xy_test1,xy_test2,xy_test3),axis=0)
        X_test = np.concatenate((X_test1,X_test2,X_test3),axis=0)
        Y_test = np.concatenate((Y_test1,Y_test2,Y_test3),axis=0)
        return X,Y,beta_true,xy,xy_test,X_test,Y_test


def one_trial_estimation_(a,T,k,ini,tol,knn_k,fix=False):
    # Generate data
    if fix>0:
        X,Y,beta_true,xy,xy_test,X_test,Y_test = data_generation_(a,T,k,seed=fix)
    else:
        X,Y,beta_true,xy,xy_test,X_test,Y_test = data_generation_(a,T,k,seed=None)
    l_boundary = [-20,-20,-20]
    u_boundary = [20,20,20]
    beta_name = ['a','b','c']
    beta_array,P_array,cluster_id,cluster_beta_0,iter_num = the_whole_model_KMeans(Y,X,beta_name,ini,l_boundary,u_boundary,tol,k,a)
    # KNN algorithm for prediction
    knn_model = KNeighborsRegressor(n_neighbors=knn_k, metric='euclidean')
    knn_model.fit(xy, beta_array)
    beta_array_test = knn_model.predict(xy_test)
    V_test = (X_test * beta_array_test[:,:,None]).sum(axis=1)
    demo = np.exp(V_test).sum(axis=1).reshape(X_test.shape[0],1)
    P_array_test = np.exp(V_test) / demo
    # Calculate metrics
    # P_array_test = P_array_test[Y_test.max(axis=1)>0.4]
    # Y_test = Y_test[Y_test.max(axis=1)>0.4]
    MAE = np.abs(P_array_test-Y_test).mean()
    OA = np.minimum(P_array_test,Y_test).sum()/Y_test.shape[0]
    SSR = np.square(P_array_test-Y_test).sum()
    SST = np.square(np.zeros(Y_test.shape)-Y_test).sum()
    ARS = 1 - SSR/SST
    return MAE, OA, ARS


# # Test
# seed = 9343
# print(one_trial_estimation_('Multimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=1,fix=seed))
# print(one_trial_estimation_('Multimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=3,fix=seed))
# print(one_trial_estimation_('Multimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=5,fix=seed))
# print(one_trial_estimation_('Multimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=1,fix=seed))
# print(one_trial_estimation_('Multimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=3,fix=seed))
# print(one_trial_estimation_('Multimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=5,fix=seed))
# print('---------')
# print(one_trial_estimation_('Unimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=1,fix=seed))
# print(one_trial_estimation_('Unimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=3,fix=seed))
# print(one_trial_estimation_('Unimodal',500,1,[-0.5,-0.5,0.5],0.1,knn_k=5,fix=seed))
# print(one_trial_estimation_('Unimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=1,fix=seed))
# print(one_trial_estimation_('Unimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=3,fix=seed))
# print(one_trial_estimation_('Unimodal',500,3,[-0.5,-0.5,0.5],0.1,knn_k=5,fix=seed))




# For tables
MAE_,OA_,ARS_ = [],[],[]
np.random.seed(8521)
random_integers = np.random.randint(1, 10001, size=20)

alpha = 'Multimodal'
T = 5000
k = 3
ini = [-0.5,-0.5,0.5]
tol = 0.1
knn_k = 5
for i in range(20):
    MAE, OA, ARS = one_trial_estimation_(alpha,T,k,ini,tol,knn_k,fix=random_integers[i])
    MAE_.append(MAE)
    OA_.append(OA)
    ARS_.append(ARS)
    
MAE_ = np.array(MAE_)
OA_ = np.array(OA_)
ARS_ = np.array(ARS_)

print(MAE_.mean(),MAE_.std())
print('-------------')
print(OA_.mean(),OA_.std())
print('-------------')
print(ARS_.mean(),ARS_.std())


















