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


def group_level_IO(Y_line,X_line,beta_0,tol=0.5,l_boundary=-10,u_boundary=10):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("mip1", env=env)
    x = m.addVars(2,lb=l_boundary, ub=u_boundary, vtype=gp.GRB.CONTINUOUS, name='x')
    x_ = m.addVars(1,lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name='x_')

    #define objective function
    obj = (x[0]-beta_0[0])**2 + (x[1]-beta_0[1])**2 + (x_[0]-beta_0[2])**2

    #define constraints
    Y_line_ = Y_line.copy()
    log_Y_line = np.log(Y_line_)
    
    V_0 = x[0]*X_line[0,0] + x[1]*X_line[1,0] +x_[0]*X_line[2,0]
    V_1 = x[0]*X_line[0,1] + x[1]*X_line[1,1] +x_[0]*X_line[2,1]
    m.addConstr(V_0-V_1 <= log_Y_line[0]-log_Y_line[1] + float(tol))
    m.addConstr(V_0-V_1 >= log_Y_line[0]-log_Y_line[1] - float(tol))
        
    m.setObjective(obj,gp.GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()

    try:
        beta = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        print('No feasible solution')
        
    return beta,Z


def one_iteration(Y,X,beta_0,tol,k=2):
    N = np.size(X,0)
    beta_array = np.zeros((N,len(beta_0)))

    for i in range(N):
        X_line = X[i,:,:]
        Y_line = Y[i,:]
        if i<6:
            beta,Z= group_level_IO(Y_line,X_line,beta_0,tol,l_boundary=-50,u_boundary=0)
        else:
            beta,Z= group_level_IO(Y_line,X_line,beta_0,tol,l_boundary=0,u_boundary=50)
        beta_array[i,:] = beta
    
    k_means = KMeans(init='random',n_clusters=k)
    k_means.fit(beta_array[:,:2])
    cluster_id = k_means.predict(beta_array[:,:2])
    cluster_beta_0 = []
    for label in range(k):
        cluster_beta_0.append(beta_array[cluster_id==label].mean(axis=0))
    cluster_beta_0 = np.array(cluster_beta_0)
    
    return (beta_array,cluster_id, cluster_beta_0)





# A test on sample data
beta_0 = np.array([0,0,0])
X = np.zeros((8,3,2))
X[0,:,:] = np.array([[10,30],[10,3],[0,1]])
X[1,:,:] = np.array([[20,40],[15,3],[0,1]])
X[2,:,:] = np.array([[40,60],[25,3],[0,1]])
X[3,:,:] = np.array([[10,30],[10,3],[0,1]])
X[4,:,:] = np.array([[20,40],[15,3],[0,1]])
X[5,:,:] = np.array([[40,60],[25,3],[0,1]])
X[6,:,:] = np.array([[10,30],[3,10],[0,1]])
X[7,:,:] = np.array([[60,10],[25,3],[0,1]])
Y = np.array([[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.1,0.9],[0.9,0.1]])



Y_line = Y[7,:]
log_Y_line = np.log(Y_line)
log_Y_line[0]-log_Y_line[1]

beta,Z = group_level_IO(Y[7,:],X[7,:,:],np.array([0,0,0]),tol=0.2,l_boundary=-100,u_boundary=100)
beta

beta_array,cluster_id, cluster_beta_0 = one_iteration(Y,X,beta_0,tol=0.05,k=3)
cluster_id

V = (X * beta_array[:,:,None]).sum(axis=1)
demo = np.exp(V).sum(axis=1).reshape(8,1)
P = np.exp(V) / demo



