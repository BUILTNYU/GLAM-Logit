# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:33:18 2022

@author: MSI-PC
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.io
from scipy import stats
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os
import pickle
from Group_level_IO import logitMLE_DIST,logitMLE, group_level_IO, compute_metrics
# from reference import logitMLE_forloop
import time
from math import radians, cos, sin, asin, sqrt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


#################
##One iteration##
#################
def one_iteration(Y,X,beta_0,tol,boundary=50):
    N = np.size(X,0)
    beta_array = np.zeros((N,len(beta_0)))
    P_array = np.zeros((N,np.size(X,2)))
    LL_0_list = []
    LL_list = []
    DIST_list = []
    mse_list = []
    mae_list = []

    for i in range(N):
        X_line = X[i,:,:]
        Y_line = Y[i,:]
        tol_ = tol
        beta = beta_0
        # try IO with specific coefficient boundary
        while beta.sum() == beta_0.sum() and tol_<=1.0:
            tol_ += 0.4
            beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol_,boundary,release=False)
        # relax that boundary
        tol_ = tol
        while beta.sum() == beta_0.sum() and tol_<=1.0:
            tol_ += 0.4
            beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol_,boundary,release=True)
            
        beta_array[i,:] = beta
        P_array[i,:] = P
        LL_0_list.append(LL_0)
        LL_list.append(LL)
        DIST_list.append(Z)
        mse_list.append(mse)
        mae_list.append(mae)
        
    LL_0_array = np.array(LL_0_list)
    LL_array = np.array(LL_list)
    DIST_array = np.array(DIST_list)
    mse_array = np.array(mse_list)
    mae_array = np.array(mae_list)
    
    good_lst = (beta_array.sum(axis=1)!=beta_0.sum())
    beta_0_new = beta_array[good_lst].mean(axis=0)
    # beta_0_new = beta_array.mean(axis=0)
    
    return (beta_array, P_array, beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array)

        

def get_feasibility_and_cluster(Y,X,tol,boundary,k):
    N = np.size(X,0)
    K = np.size(X,1)
    beta_0 = np.zeros(K)
    beta_array = np.zeros((N,K))
    LL_0_sum = 0
    LL_sum = 0
    feasible_list = []
    
    for i in range(N):
        X_line = X[i,:,:]
        Y_line = Y[i,:]
        beta = beta_0
        # try IO with specific coefficient boundary
        tol_ = tol
        while beta.sum() == beta_0.sum() and tol_<=1.5:
            tol_ += 0.2
            beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol_,boundary,release=False)
        if beta.sum() != beta_0.sum():
            feasible_list.append(('negative',round(tol_,2)))
            beta_array[i,:] = beta
            LL_0_sum += LL_0
            LL_sum += LL
            
            continue
        # relax that boundary
        tol_ = tol
        while beta.sum() == beta_0.sum() and tol_<=1.5:
            tol_ += 0.2
            beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol_,boundary,release=True)
        if beta.sum() != beta_0.sum():
            feasible_list.append(('non-negative',round(tol_,2)))
            beta_array[i,:] = beta
            LL_0_sum += LL_0
            LL_sum += LL
        else:
            feasible_list.append(('infeasible',999))
            LL_0_sum += LL_0
            LL_sum += LL_0
    
    # get feasible array
    feasible_array = np.array(feasible_list)
    # use K-means to calcualte multi fixed points
    k_means = KMeans(init="k-means++", n_clusters=k)
    k_means.fit(beta_array[feasible_array[:,0]!='infeasible'])
    cluster_id = k_means.predict(beta_array)
    cluster_id[feasible_array[:,0]=='infeasible'] = 999
    cluster_beta_0 = []
    for label in range(k):
        cluster_beta_0.append(beta_array[cluster_id==label].mean(axis=0))
    cluster_beta_0 = np.array(cluster_beta_0)
    
    return feasible_array,cluster_id,cluster_beta_0,LL_0_sum,LL_sum


def one_iteration_KMeans(Y,X,feasible_array,cluster_id,cluster_beta_0,boundary,k):
    N = np.size(X,0)
    K = np.size(X,1)
    beta_array = np.zeros((N,K))
    P_array = np.zeros((N,np.size(X,2)))
    LL_0_list = []
    LL_list = []
    DIST_list = []
    mse_list = []
    mae_list = []
    cluster_id_new = []
    # for loop each agent
    for i in range(N):
        X_line = X[i,:,:]
        Y_line = Y[i,:]
        # for those bad agents, fill with fixed point
        if feasible_array[i,0] == 'infeasible':
            LL_best = -999
            beta_best = np.zeros(K)
            label_best = 999
            for label in range(k):
                beta = cluster_beta_0[label]
                rho,mse,mae,LL,LL_0,P = compute_metrics(Y_line,X_line,beta)
                if LL > LL_best and LL > LL_0:
                    beta_best = beta
                    label_best = label
                    
            rho,mse,mae,LL,LL_0,P = compute_metrics(Y_line,X_line,beta_best)
            beta_array[i,:] = beta_best
            P_array[i,:] = P
            LL_0_list.append(LL_0)
            LL_list.append(LL)
            DIST_list.append(0)
            mse_list.append(mse)
            mae_list.append(mae)
            cluster_id_new.append(label_best)
            continue
        # for those good agents, update k fixed points
        cluster = int(cluster_id[i])
        beta_0 = cluster_beta_0[cluster]
        release = False if feasible_array[i,0]=='negative' else True
        tol_ = float(feasible_array[i,1])
        beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol_,boundary,release)
        beta_array[i,:] = beta
        P_array[i,:] = P
        LL_0_list.append(LL_0)
        LL_list.append(LL)
        DIST_list.append(Z)
        mse_list.append(mse)
        mae_list.append(mae)
        cluster_id_new.append(cluster)
        
    LL_0_array = np.array(LL_0_list)
    LL_array = np.array(LL_list)
    DIST_array = np.array(DIST_list)
    mse_array = np.array(mse_list)
    mae_array = np.array(mae_list)
    cluster_id_new = np.array(cluster_id_new)
    cluster_beta_0_new = []
    for label in range(k):
        cluster_beta_0_new.append(beta_array[cluster_id_new==label].mean(axis=0))
    cluster_beta_0_new = np.array(cluster_beta_0_new)
    
    return (beta_array, P_array, cluster_id_new, cluster_beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array)



###################
##The whole model##
###################
def the_whole_model(Y,X,beta_0,tol,boundary):
    iter_num = 0
    beta_0_dict = {}
    LL_dict = {}
    DIST_dict = {}
    MSE_dict = {}
    MAE_dict = {}

    # initialize
    beta_array, P_array, beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array = one_iteration(Y,X,beta_0,tol,boundary)
    beta_0_dict[0] = beta_0
    LL_dict[0] = LL_0_array
    DIST_dict[0] = 0
    MSE_dict[0] = 0
    MAE_dict[0] = 0
    beta_0_dict[1] = beta_0_new
    LL_dict[1] = LL_array
    DIST_dict[1] = DIST_array
    MSE_dict[0] = mse_array
    MAE_dict[0] = mae_array

    iter_num += 1
    beta_0 = beta_0_new
    change = (beta_0_dict[1]-beta_0_dict[0])/(beta_0_dict[0]+1e-6)

    #do iteration
    while np.sum(np.abs(change))/len(beta_0)>=0.005 and iter_num<=1000:
        iter_num += 1
        beta_array, P_array, beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array = one_iteration(Y,X,beta_0,tol,boundary)
        beta_0 = (iter_num-1)/iter_num * beta_0 + 1/iter_num * beta_0_new
        beta_0_dict[iter_num] = beta_0
        LL_dict[iter_num] = LL_array
        DIST_dict[iter_num] = DIST_array
        MSE_dict[iter_num] = mse_array
        MAE_dict[iter_num] = mae_array
        change = (beta_0_dict[iter_num]-beta_0_dict[iter_num-1])/(beta_0_dict[iter_num-1]+1e-6)
        if iter_num%5 == 0:
            print ('finished %i iterations'%iter_num)
            print ('percent change: %.4f'%(np.sum(np.abs(change))/len(beta_0)))
            print ('LL value:%.2f'%LL_array.sum())
            print ('VOT: %.2f'%(beta_array[:,0]/beta_array[:,6]).mean())
    
    return beta_array,P_array,beta_0_dict,LL_dict,DIST_dict,MSE_dict,MAE_dict

def the_whole_model_KMeans(Y,X,beta_name,tol,boundary,k):
    N = np.size(X,0)
    K = np.size(X,1)
    cluster_beta_0_dict = {}
    LL_dict = {}
    DIST_dict = {}
    MSE_dict = {}
    MAE_dict = {}
    # initialize
    iter_num = 0
    cluster_beta_0_dict[0] = np.array([np.zeros(K)]*k)
    DIST_dict[0] = np.zeros(N)
    MSE_dict[0] = np.zeros(N)
    MAE_dict[0] = np.zeros(N)
    # first iteration
    iter_num += 1
    feasible_array,cluster_id,cluster_beta_0,LL_0_sum,LL_sum = get_feasibility_and_cluster(Y,X,tol,boundary,k)
    beta_array, P_array, cluster_id_new, cluster_beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array = one_iteration_KMeans(Y,X,feasible_array,cluster_id,cluster_beta_0,boundary,k)
    LL_dict[0] = LL_0_array # here pay attention to LL_0
    cluster_beta_0_dict[1] = cluster_beta_0_new
    LL_dict[1] = LL_array
    DIST_dict[1] = DIST_array
    MSE_dict[1] = mse_array
    MAE_dict[1] = mae_array
    cluster_id = cluster_id_new
    cluster_beta_0 = cluster_beta_0_dict[iter_num]
    change = (cluster_beta_0_dict[1]-cluster_beta_0_dict[0])/(cluster_beta_0_dict[0]+1e-6)
    print('first iteration finished')

    #do the rest iterations
    while np.mean(np.abs(change))>=0.005 and iter_num<=1000:
        iter_num += 1
        beta_array, P_array, cluster_id_new, cluster_beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array = one_iteration_KMeans(Y,X,feasible_array,cluster_id,cluster_beta_0,boundary,k)
        updated_fixed_point = (iter_num-1)/iter_num * cluster_beta_0_dict[iter_num-1] + 1/iter_num * cluster_beta_0_new
        cluster_beta_0_dict[iter_num] = updated_fixed_point
        LL_dict[iter_num] = LL_array
        DIST_dict[iter_num] = DIST_array
        MSE_dict[iter_num] = mse_array
        MAE_dict[iter_num] = mae_array
        change = (cluster_beta_0_dict[iter_num]-cluster_beta_0_dict[iter_num-1])/(cluster_beta_0_dict[iter_num-1]+1e-6)
        cluster_id = cluster_id_new
        cluster_beta_0 = cluster_beta_0_dict[iter_num]
        print ('finished %i iterations'%iter_num)
        print ('percent change: %.4f'%(np.mean(np.abs(change))))
        print ('LL value:%.2f'%LL_array.sum())
    
    return beta_array,P_array,cluster_id_new,cluster_beta_0_dict,LL_dict,DIST_dict,MSE_dict,MAE_dict,feasible_array




##################
##Model building##
##################    
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\X_all.pickle", 'rb') as handle:
    X = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\Y_all.pickle", 'rb') as handle:
    Y = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\num_all.pickle", 'rb') as handle:
    num = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\id_all.pickle", 'rb') as handle:
    group_id = pickle.load(handle)


beta_0 = np.array([0,0,0,0,0,0,0,0,0,0])
beta_name = ['auto_tt','transit_ivt','transit_at','transit_et','transit_nt','non_vehicle_tt',
              'cost','constant_auto','constant_transit','constant_non_vehicle']

# one iteration
start_time = time.time()
feasible_array,cluster_id,cluster_beta_0,LL_0_sum,LL_sum = get_feasibility_and_cluster(Y,X,tol=0.1,boundary=50,k=2)
end_time = time.time()

print('Computational time: %.4f seconds'%(end_time-start_time))
print(sum(cluster_id==0))
print(sum(cluster_id==1))
print('percent of good agents: %.3f'%(sum(feasible_array[:,0]!='infeasible')/len(feasible_array)))
print('rho: %.4f'%(1-LL_sum/LL_0_sum))

# with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\feasible_array.pickle', 'wb') as handle:
#     pickle.dump(feasible_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\cluster_id.pickle', 'wb') as handle:
#     pickle.dump(cluster_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\cluster_beta_0.pickle', 'wb') as handle:
#     pickle.dump(cluster_beta_0, handle, protocol=pickle.HIGHEST_PROTOCOL)  

start_time = time.time()
beta_array, P_array, cluster_id_new, cluster_beta_0_new, LL_0_array, LL_array, DIST_array,mse_array,mae_array = one_iteration_KMeans(Y,X,feasible_array,cluster_id,cluster_beta_0,boundary=50,k=2)
end_time = time.time()

print('Computational time: %.4f seconds'%(end_time-start_time))
num_fixed_point = sum((beta_array.sum(axis=1)==cluster_beta_0[0].sum())|(beta_array.sum(axis=1)==cluster_beta_0[1].sum()))
num_zero = sum(beta_array.sum(axis=1)==0)
num_good = sum((beta_array.sum(axis=1)!=cluster_beta_0[0].sum())&\
               (beta_array.sum(axis=1)!=cluster_beta_0[1].sum())&\
                   (beta_array.sum(axis=1)!=0))
print('percent of good agents: %.3f'%(num_good/len(beta_array)))
print('percent of fixed-point agents: %.3f'%(num_fixed_point/len(beta_array)))
print('percent of zero agents: %.3f'%(num_zero/len(beta_array)))
print('rho: %.4f'%(1-LL_array.sum()/LL_0_array.sum()))
for label in range(len(cluster_beta_0_new)):
    aa = sum((cluster_id_new==label)&(feasible_array[:,0]!='infeasible'))
    bb = sum(feasible_array[:,0]!='infeasible')
    print('percent of cluster %i in good agents: %.3f'%(label,aa/bb))







###############
# whole model #
###############
# start_time = time.time()
# beta_array,P_array,beta_0_dict,LL_dict,DIST_dict,MSE_dict,MAE_dict = the_whole_model(Y[100000:100500,:],X[100000:100500,:,:],beta_0,tol=0.2,boundary=50)
# end_time = time.time()
# print('Computational time: %.4f seconds'%(end_time-start_time))
# # KPI
# beta_final = beta_0_dict[max(beta_0_dict.keys())-1]
# print('Value of time: $%.2f/hour'% (beta_final[0]/beta_final[6]).mean())
# good_lst = beta_array.sum(axis=1)!=beta_final.sum()
# print('precent of good agent: %.3f'%(good_lst.sum()/len(beta_array)))
# print('rho:%.4f' %(1 - LL_dict[max(LL_dict.keys())].sum()/LL_dict[0].sum()))
# MAE_dict[max(LL_dict.keys())].mean()


# KMeans model
start_time = time.time()
beta_array,P_array,cluster_id_new,cluster_beta_0_dict,LL_dict,DIST_dict,MSE_dict,MAE_dict,feasible_array = the_whole_model_KMeans(Y,X,beta_name,tol=0.2,boundary=50,k=5)
end_time = time.time()

# KPI
print('Computational time: %.4f seconds'%(end_time-start_time))
cluster_beta_final = cluster_beta_0_dict[max(cluster_beta_0_dict.keys())]
cluster_beta_check = cluster_beta_0_dict[max(cluster_beta_0_dict.keys())-1]

aa = np.zeros(len(beta_array))>0
for i in range(len(cluster_beta_check)):
    aa = aa | (beta_array.sum(axis=1)==cluster_beta_check[i].sum())
num_fixed_point = sum(aa)
num_zero = sum(beta_array.sum(axis=1)==0)
bb = np.zeros(len(beta_array))==0
for i in range(len(cluster_beta_check)):
    bb = bb & (beta_array.sum(axis=1)!=cluster_beta_check[i].sum())
bb = bb & (beta_array.sum(axis=1)!=0)
num_good = sum(bb)
print('percent of good agents: %.3f'%(num_good/len(beta_array)))
print('percent of fixed-point agents: %.3f'%(num_fixed_point/len(beta_array)))
print('percent of zero agents: %.3f'%(num_zero/len(beta_array)))
print('rho: %.4f'%(1-LL_dict[max(LL_dict.keys())].sum()/LL_dict[0].sum()))
print('MAE: %.4f'%MAE_dict[max(MAE_dict.keys())].mean())
for label in range(len(cluster_beta_check)):
    cc = sum((cluster_id_new==label)&(feasible_array[:,0]!='infeasible'))
    dd = sum(feasible_array[:,0]!='infeasible')
    print('percent of cluster %i in good agents: %.3f'%(label,cc/dd))
for label in range(len(cluster_beta_final)):
    vot = cluster_beta_final[label,0]/cluster_beta_final[label,6]
    num = sum((cluster_id_new==label)&(feasible_array[:,0]!='infeasible'))
    print('cluster %i VOT: %.2f$/hour, number of good agents %i'%(label,vot,num))

# VOT
def get_VOT(df):
    vot = df[0]/df[6]
    vot[vot>=100] = 50
    vot[vot<=-100] = -50
    return vot.mean()



# 4-group in NY
marker = (cluster_id_new==0)&(feasible_array[:,0]!='infeasible')
beta_result = pd.DataFrame(beta_array[marker])
beta_result['group_id'] = group_id[marker]
aa = beta_result[beta_result['group_id'].map(lambda x: x.split('_')[-1])=='NotLowIncome']
bb = beta_result[beta_result['group_id'].map(lambda x: x.split('_')[-1])=='LowIncome']
cc = beta_result[beta_result['group_id'].map(lambda x: x.split('_')[-1])=='Senior']
dd = beta_result[beta_result['group_id'].map(lambda x: x.split('_')[-1])=='Student']
print('NY state NotLowIncome VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(aa),len(aa)))
print('NY state LowIncome VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(bb),len(bb)))
print('NY state Senior VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(cc),len(cc)))
print('NY state Student VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(dd),len(dd)))

# 4-group in NYC
NYC_county = ['36061','36047','36005','36081','36085']
region_marker = beta_result['group_id'].map(lambda x: (x.split('_')[0][:5] in NYC_county) & (x.split('_')[1][:5] in NYC_county))
aa = beta_result[(beta_result['group_id'].map(lambda x: x.split('_')[-1])=='NotLowIncome')&region_marker]
bb = beta_result[(beta_result['group_id'].map(lambda x: x.split('_')[-1])=='LowIncome')&region_marker]
cc = beta_result[(beta_result['group_id'].map(lambda x: x.split('_')[-1])=='Senior')&region_marker]
dd = beta_result[(beta_result['group_id'].map(lambda x: x.split('_')[-1])=='Student')&region_marker]
print('NYC NotLowIncome VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(aa),len(aa)))
print('NYC LowIncome VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(bb),len(bb)))
print('NYC Senior VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(cc),len(cc)))
print('NYC Student VOT: %.2f$/hour, number of good agents: %i'%(get_VOT(dd),len(dd)))









cluster_beta_0_result = pd.concat([pd.DataFrame(values) for values in cluster_beta_0_dict.values()])
beta_0_result = pd.DataFrame(cluster_beta_0_result.loc[0,:].values)
fig,ax = plt.subplots()
beta_0_result[0].plot(ax = ax,label=beta_name[0])
beta_0_result[1].plot(ax = ax,label=beta_name[1])
beta_0_result[2].plot(ax = ax,label=beta_name[2])
beta_0_result[3].plot(ax = ax,label=beta_name[3])
beta_0_result[4].plot(ax = ax,label=beta_name[4])
beta_0_result[5].plot(ax = ax,label=beta_name[5])
plt.title('Coefficient of travel time in each iteration')
plt.legend()
plt.xlabel('iteration_num')
plt.ylabel('value')

beta_result = pd.DataFrame(beta_array)
fig,ax = plt.subplots()
beta_result[0].plot.kde(ax=ax,label=beta_name[0])

fig,ax = plt.subplots()
beta_result[0].hist(ax = ax,bins=100,label=beta_name[0],alpha=0.8)
beta_result[1].hist(ax = ax,bins=30,label=beta_name[1],alpha=0.8)
beta_result[2].hist(ax = ax,bins=20,label=beta_name[2],alpha=0.8)
beta_result[3].hist(ax = ax,bins=20,label=beta_name[3],alpha=0.8)
beta_result[4].hist(ax = ax,bins=30,label=beta_name[4],alpha=0.8)
beta_result[5].hist(ax = ax,bins=100,label=beta_name[5],alpha=0.8)
plt.xlim([-20,20])
plt.legend(loc='upper left')
plt.title('Distribution of theta_i after the final iteration')
plt.xlabel('value')
plt.ylabel('frequency')



path = 'k5'
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\beta_array.pickle'%path, 'wb') as handle:
    pickle.dump(beta_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\P_array.pickle'%path, 'wb') as handle:
    pickle.dump(P_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\cluster_id_new.pickle'%path, 'wb') as handle:
    pickle.dump(cluster_id_new, handle, protocol=pickle.HIGHEST_PROTOCOL)   
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\cluster_beta_0_dict.pickle'%path, 'wb') as handle:
    pickle.dump(cluster_beta_0_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\LL_dict.pickle'%path, 'wb') as handle:
    pickle.dump(LL_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)     
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\DIST_dict.pickle'%path, 'wb') as handle:
    pickle.dump(DIST_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)     
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\MSE_dict.pickle'%path, 'wb') as handle:
    pickle.dump(MSE_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)     
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\MAE_dict.pickle'%path, 'wb') as handle:
    pickle.dump(MAE_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)     
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\feasible_array.pickle'%path, 'wb') as handle:
    pickle.dump(feasible_array, handle, protocol=pickle.HIGHEST_PROTOCOL)       
    





###################
##Result Analysis##
###################
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\beta_array.pickle', 'rb') as handle:
    beta_array = pickle.load(handle)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\P_array.pickle', 'rb') as handle:
    P_array = pickle.load(handle)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\cluster_id_new.pickle', 'rb') as handle:
    cluster_id_new = pickle.load(handle)  
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\cluster_beta_0_dict.pickle', 'rb') as handle:
    cluster_beta_0_dict = pickle.load(handle) 
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\LL_dict.pickle', 'rb') as handle:
    LL_dict = pickle.load(handle)    
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\DIST_dict.pickle', 'rb') as handle:
    DIST_dict = pickle.load(handle) 
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\MSE_dict.pickle', 'rb') as handle:
    MSE_dict = pickle.load(handle)  
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\MAE_dict.pickle', 'rb') as handle:
    MAE_dict = pickle.load(handle)


print('rho:%.4f' %(1 - LL_dict[max(LL_dict.keys())][50000:100000].sum()/LL_dict[0][50000:100000].sum()))




# MNL comparison
beta_dict,LL_dict,P,Hessian_dict,Jacobian_dict,model_summary = logitMLE(Y, X, tol=0.001)
print((1-LL_dict[max(LL_dict.keys())]/LL_dict[0]))
beta_final = beta_dict[max(beta_dict.keys())]
print('Value of time: $%.2f/hour'% (beta_final[0]/beta_final[6]))



# coefficient distribution
pd.Series(LL_dict.values()).plot()
plt.xlabel('iteration_num')
plt.ylabel('Log-likelihood value')
plt.title('Log-likelihood value after each iteration')

beta_0_result = pd.DataFrame(np.row_stack(beta_0_dict.values()))
fig,ax = plt.subplots()
beta_0_result[0].plot(ax = ax,label=beta_name[0])
beta_0_result[1].plot(ax = ax,label=beta_name[1])
beta_0_result[2].plot(ax = ax,label=beta_name[2])
beta_0_result[3].plot(ax = ax,label=beta_name[3])
beta_0_result[4].plot(ax = ax,label=beta_name[4])
beta_0_result[5].plot(ax = ax,label=beta_name[5])
plt.title('Coefficient of travel time in each iteration')
plt.legend()
plt.xlabel('iteration_num')
plt.ylabel('value')

fig,ax = plt.subplots()
beta_0_result[6].plot(ax = ax,label=beta_name[6])
beta_0_result[7].plot(ax = ax,label=beta_name[7])
beta_0_result[8].plot(ax = ax,label=beta_name[8])
beta_0_result[9].plot(ax = ax,label=beta_name[9])
plt.title('Coefficient of trip fare and mode constant in each iteration')
plt.legend()
plt.xlabel('iteration_num')
plt.ylabel('value')




beta_result = pd.DataFrame(beta_array)
fig,ax = plt.subplots()
beta_result[0].hist(ax = ax,bins=80,label=beta_name[0],alpha=0.8)
beta_result[1].hist(ax = ax,bins=30,label=beta_name[1],alpha=0.8)
beta_result[2].hist(ax = ax,bins=30,label=beta_name[2],alpha=0.8)
beta_result[3].hist(ax = ax,bins=30,label=beta_name[3],alpha=0.8)
beta_result[4].hist(ax = ax,bins=30,label=beta_name[4],alpha=0.8)
beta_result[5].hist(ax = ax,bins=80,label=beta_name[5],alpha=0.8)
plt.xlim([-20,20])
plt.legend(loc='upper left')
plt.title('Distribution of theta_i after the final iteration')
plt.xlabel('value')
plt.ylabel('frequency')

fig,ax = plt.subplots()
beta_result[6].hist(ax = ax,bins=100,label=beta_name[6],alpha=0.8)
beta_result[7].hist(ax = ax,bins=100,label=beta_name[7],alpha=0.8)
beta_result[8].hist(ax = ax,bins=30,label=beta_name[8],alpha=0.8)
beta_result[9].hist(ax = ax,bins=100,label=beta_name[9],alpha=0.8)
plt.xlim([-10,10])
plt.legend(loc='upper left')
plt.title('Distribution of theta_i after the final iteration')
plt.xlabel('value')
plt.ylabel('frequency')



# spatial characteristics
data = pd.read_csv('NYC_Processed_OD_Student.csv')[['origin_bgrp', 'destination_bgrp', 'lng_o', 'lat_o', 'lng_d', 'lat_d', 'Trip_num']]

def geodistance(record):
    lng1,lat1,lng2,lat2 = map(radians,[float(record['lng_o']), float(record['lat_o']), float(record['lng_d']), float(record['lat_d'])])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    distance=2*asin(sqrt(a))*6371*1000/1609.34
    distance=round(distance,4)
    return distance

data['line_distance'] = data.apply(geodistance,axis=1)

data['Duration'] = beta[:,0]
data['Cost'] = beta[:,1]
data['Private_auto'] = beta[:,2]
data['Public_transit'] = beta[:,3]
data['Walking'] = beta[:,4]
data['Biking'] = beta[:,5]
data['On_demand_auto'] = beta[:,6]


from shapely.geometry import Point, LineString

geometry_O = [Point(xy) for xy in zip(data.lng_o, data.lat_o)]
geometry_D = [Point(xy) for xy in zip(data.lng_d, data.lat_d)]
OD_lst = []
for i in range(len(data)):
    OD = LineString([geometry_O[i],geometry_D[i]])
    OD_lst.append(OD)

g = gpd.GeoSeries(OD_lst)
data = gpd.GeoDataFrame(data,geometry=g,crs="EPSG:4326")
# data.to_file(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\Student\OD_results.shp')

NotLowIncome = gpd.read_file(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\NotLowIncome\OD_results.shp')
LowIncome = gpd.read_file(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\LowIncome\OD_results.shp')
Senior = gpd.read_file(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\Senior\OD_results.shp')
Student = gpd.read_file(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\Student\OD_results.shp')

fig,ax = plt.subplots(1,1,figsize=(12,12))
Student.plot(linewidth=0.1)
plt.axis('off')
# plt.savefig('C:/Users/MSI-PC/Desktop/Student.jpg',
#             dpi=300,
#             bbox_inches = 'tight',
#             facecolor = 'w',
#             edgecolor = 'w')



#####################
##Model Prediection##
#####################
with open('X_LowIncome.pickle', 'rb') as handle:
    X = pickle.load(handle)
with open('Y_LowIncome.pickle', 'rb') as handle:
    Y = pickle.load(handle)
with open('num_LowIncome.pickle', 'rb') as handle:
    num = pickle.load(handle)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\model_results\LowIncome\beta.pickle', 'rb') as handle:
    beta = pickle.load(handle)

pricing_area = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\NYC region\Congestion_pricing_area.shp")


# pick out OD links
lst = []
for i in range(len(LowIncome)):
    ct2010_o = str(LowIncome['origin_bgr'].iloc[i])[5:-1]
    ct2010_d = str(LowIncome['destinatio'].iloc[i])[5:-1]
    if ct2010_d in list(pricing_area['ct2010']) and ct2010_o not in list(pricing_area['ct2010']):
        lst.append(i) 

X = X[lst,:,:]
beta = beta[lst,:]
num = num[lst]

# calculate mode share given X and beta and num
V = (X * beta[:,:,None]).sum(axis=1)
demo = np.exp(V).sum(axis=1).reshape(np.size(X,0),1)
P = np.exp(V) / demo
mode_share = ((P * num[:,None])/num.sum()).sum(axis=0)

# add congestion pricing
X[:,1,0] += 15  # cost, private_auto, plus $5
X[:,1,4] += 15  # cost, on_demand, plus $5
        
# recalculate mode share given new X and beta and num
V2 = (X * beta[:,:,None]).sum(axis=1)
demo2 = np.exp(V2).sum(axis=1).reshape(np.size(X,0),1)
P2 = np.exp(V2) / demo2
mode_share2 = ((P2 * num[:,None])/num.sum()).sum(axis=0)









