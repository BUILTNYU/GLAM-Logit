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

# assume only linear MNL for now, not other GEV
# beta is the vector of beta's being estimated
# Y is an N x J matrix, with N data samples and J alternatives
# X is an N x (K+J-1) x J matrix, with K variables and J-1 constants
def compute_elements(Y,X,N,K_,NumAlt,beta,beta_0,lambda_):
    # compute the probabilities for the alternatives for each observation n, Pn(i)
    V = (X * beta.reshape(1,K_,1)).sum(axis=1)
    demo = np.exp(V).sum(axis=1).reshape(N,1)
    P = np.exp(V) / demo

    # compute the initial log-likelihood function (LL)
    LL = np.sum(Y * np.log(P))
    
    # compute DIST which is (beta_0 - beta)**2
    DIST = ((beta - beta_0)**2).sum()

    # weighted objective function
    Objective = (1-lambda_)*DIST + lambda_*-LL

    # compute Jacobian
    Jacobian_LL = (X * (Y[:,None,:]-P[:,None,:])).sum(axis=2).sum(axis=0)
    Jacobian_DIST = 2*beta - 2*beta_0
    Jacobian = (1-lambda_)*Jacobian_DIST + lambda_*-Jacobian_LL

    # compute Hessian
    Hessian_LL = np.zeros((K_,K_))
    for k in range(K_):
        for l in range(K_):
            k_sum = (X[:,k,:][:,None,:] * P[:,None,:]).sum(axis=2).squeeze()
            l_sum = (X[:,l,:][:,None,:] * P[:,None,:]).sum(axis=2).squeeze()
            Hessian_LL[k,l] = np.sum(-P * (X[:,k,:] - k_sum[:,None]) * (X[:,l,:] - l_sum[:,None]))
    Hessian_DIST = np.diag(np.ones(K_)*2)
    Hessian = (1-lambda_)*Hessian_DIST + lambda_*-Hessian_LL
    
    return (P,LL,DIST,Objective,Jacobian,Hessian)



def logitMLE(Y, X, tol):
    NumAlt = np.size(Y,1) # number of alternatives
    N = np.size(Y,0)  # number of individuals
    beta_dict = {}
    LL_dict = {}
    Hessian_dict = {}
    Jacobian_dict = {}
    
    # initiate the beta's
    beta_name = ['constant','var_1']
    beta = np.zeros(np.size(X,1)) # columns represent each iteration
    beta_dict[0] = beta
    K_ = np.size(beta,0) # Here I prefer to use K_ = K+J-1 do differeniate from K
    
    def compute_elements(N,K_,NumAlt,beta):
        # compute the probabilities for the alternatives for each observation n, Pn(i)
        V = (X * beta.reshape(1,K_,1)).sum(axis=1)
        demo = np.exp(V).sum(axis=1).reshape(N,1)
        P = np.exp(V) / demo
    
        # compute the initial log-likelihood function (LL)
        LL = np.sum(Y * np.log(P))
        
        # compute Jacobian
        Jacobian = (X * (Y[:,None,:]-P[:,None,:])).sum(axis=2).sum(axis=0)
    
        # compute Hessian
        Hessian = np.zeros((K_,K_))
        for k in range(K_):
            for l in range(K_):
                k_sum = (X[:,k,:][:,None,:] * P[:,None,:]).sum(axis=2).squeeze()
                l_sum = (X[:,l,:][:,None,:] * P[:,None,:]).sum(axis=2).squeeze()
                Hessian[k,l] = np.sum(-P * (X[:,k,:] - k_sum[:,None]) * (X[:,l,:] - l_sum[:,None]))
        
        return (P,LL,Jacobian,Hessian)

    # run Newton-Raphson algorithm
    deltabeta = tol*1.01
    maxdeltabeta = tol*1.01
    iter_num = 0
    
    P,LL,Jacobian,Hessian = compute_elements(N,K_,NumAlt,beta)
    beta_dict[0] = beta
    LL_dict[0] = LL
    Hessian_dict[0] = Hessian
    Jacobian_dict[0] = Jacobian

    while deltabeta > tol and maxdeltabeta > tol and iter_num <= 10000:
        # we can treat LL' as a new function and use Newton's method to find roots
        # beta_new = beta - Jocabian/Hessian
        iter_num += 1
        beta_dict[iter_num] = beta_dict[iter_num-1] - np.dot(np.linalg.inv(Hessian), Jacobian) 
        P,LL,Jacobian,Hessian = compute_elements(N,K_,NumAlt,beta_dict[iter_num])
        LL_dict[iter_num] = LL
        Hessian_dict[iter_num] = Hessian
        Jacobian_dict[iter_num] = Jacobian
    
        deltabeta = np.sqrt(np.sum((beta_dict[iter_num] - beta_dict[iter_num-1])**2)/K_)
        maxdeltabeta = max(np.abs((beta_dict[iter_num]-beta_dict[iter_num-1])/(beta_dict[iter_num-1]+1e-8)))
        
    # model performance measures
    rho = 1 - LL_dict[iter_num]/LL_dict[0]
    rhobar2 = 1 - (LL_dict[iter_num]-K_)/LL_dict[0]
    Pc = Y.sum(axis=0)/N  #probability estimated model with constant
    LLc = np.sum(Y * np.log(Pc.reshape(1,NumAlt)))
    Chi0 = -2 * (LL_dict[0] - LL_dict[iter_num])
    Chic = -2 * (LLc - LL_dict[iter_num])
    rho2 = 1 - LL_dict[iter_num]/LLc
    Covbeta = -np.linalg.inv(Hessian_dict[iter_num])
    betastd = np.sqrt(np.diagonal(Covbeta))
    tstat = beta_dict[iter_num]/betastd
    pval = stats.t.sf(np.abs(tstat), N-1)*2
    
    model_summary = {'rho':rho, 'rho2':rho2, 'LLc':LLc ,'rhobar2':rhobar2, 'Chi0':Chi0, 'Chic':Chic,
                     'var_name':beta_name,'coefficients':beta_dict[iter_num], 
                     'std':betastd, 'tstat':tstat,'pval':pval}
    
    return (beta_dict,LL_dict,P,Hessian_dict,Jacobian_dict,model_summary)

def logitMLE_DIST(Y, X, tol, beta_0, lambda_):
    NumAlt = np.size(Y,1) # number of alternatives
    N = np.size(Y,0)  # number of individuals
    beta_dict = {}
    P_dict = {}
    LL_dict = {}
    DIST_dict = {}
    Objective_dict = {}
    Hessian_dict = {}
    Jacobian_dict = {}
    
    # initiate the beta's
    beta = beta_0 # columns represent each iteration
    beta_dict[0] = beta
    K_ = np.size(beta,0) # Here I prefer to use K_ = K+J-1 do differeniate from K
    
    # run Newton-Raphson algorithm
    deltabeta = tol*1.01
    maxdeltabeta = tol*1.01
    iter_num = 0
    
    P,LL,DIST,Objective,Jacobian,Hessian = compute_elements(Y,X,N,K_,NumAlt,beta,beta_0,lambda_)
    beta_dict[0] = beta
    P_dict[0] = P
    LL_dict[0] = LL
    DIST_dict[0] = DIST
    Objective_dict[0] = Objective
    Hessian_dict[0] = Hessian
    Jacobian_dict[0] = Jacobian
    
    while deltabeta > tol and maxdeltabeta > tol and iter_num <= 100:
        # we can treat LL' as a new function and use Newton's method to find roots
        # beta_new = beta - Jocabian/Hessian
        delta = np.dot(np.linalg.inv(Hessian), Jacobian)
        beta = beta - delta
        iter_num += 1
        P,LL,DIST,Objective,Jacobian,Hessian = compute_elements(Y,X,N,K_,NumAlt,beta,beta_0,lambda_)

        beta_dict[iter_num] = beta
        LL_dict[iter_num] = LL
        P_dict[iter_num] = P
        DIST_dict[iter_num] = DIST
        Objective_dict[iter_num] = Objective
        Hessian_dict[iter_num] = Hessian
        Jacobian_dict[iter_num] = Jacobian
            
        deltabeta = np.sqrt(np.sum((beta_dict[iter_num] - beta_dict[iter_num-1])**2)/K_)
        maxdeltabeta = max(np.abs((beta_dict[iter_num]-beta_dict[iter_num-1])/(beta_dict[iter_num-1]+1e-6)))
    
    return (beta_dict,LL_dict,DIST_dict,Objective_dict,P)




def compute_metrics(Y_line,X_line,beta):
    V = (X_line*beta[:,None]).sum(axis=0)
    demo = np.exp(V).sum()
    P = np.exp(V) / demo
    LL = np.sum(Y_line * np.log(P))
    P_0 = np.zeros(6)+1/6
    LL_0 = np.sum(Y_line * np.log(P_0))
    rho = 1-LL/LL_0
    mse = np.mean(np.square(P-Y_line))
    mae = np.mean(np.abs(P-Y_line))
    return rho,mse,mae,LL,LL_0,P

def group_level_IO(Y_line,X_line,beta_0,tol=0.5,boundary=50,release=False):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("mip1", env=env)
    if release == False:
        x_negative = m.addVars(3,lb=-boundary, ub=0, vtype=gp.GRB.CONTINUOUS, name='x')
    else:
        x_negative = m.addVars(3,lb=-boundary, ub=boundary, vtype=gp.GRB.CONTINUOUS, name='x')
    x_ = m.addVars(7,lb=-boundary, ub=boundary, vtype=gp.GRB.CONTINUOUS, name='x_')

    #define objective function
    obj = (x_negative[0]-beta_0[0])**2 + (x_negative[1]-beta_0[1])**2 + (x_negative[2]-beta_0[2])**2 +\
          (x_[0]-beta_0[3])**2 + (x_[1]-beta_0[4])**2 + (x_[2]-beta_0[5])**2 +\
          (x_[3]-beta_0[6])**2 + (x_[4]-beta_0[7])**2 + (x_[5]-beta_0[8])**2 +\
          (x_[6]-beta_0[9])**2

    #define constraints
    Y_line_ = Y_line.copy()
    Y_line_[Y_line_<0.001] = 0.001
    log_Y_line = np.log(Y_line_)

    for j in range(6):
        V_j = x_negative[0]*X_line[0,j] + x_negative[1]*X_line[1,j] + x_negative[2]*X_line[6,j] +\
              x_[0]*X_line[2,j] + x_[1]*X_line[3,j] + x_[2]*X_line[4,j] + x_[3]*X_line[5,j] +\
              x_[4]*X_line[7,j] + x_[5]*X_line[8,j] + x_[6]*X_line[9,j]
        for k in range(6):
            V_k = x_negative[0]*X_line[0,k] + x_negative[1]*X_line[1,k] + x_negative[2]*X_line[6,k] +\
                  x_[0]*X_line[2,k] + x_[1]*X_line[3,k] + x_[2]*X_line[4,k] + x_[3]*X_line[5,k] +\
                  x_[4]*X_line[7,k] + x_[5]*X_line[8,k] + x_[6]*X_line[9,k]
            if (Y_line_[j]>=0.001 or Y_line_[k]>=0.001) and j!=k:
                m.addConstr(V_j-V_k <= log_Y_line[j]-log_Y_line[k] + float(tol))
                m.addConstr(V_j-V_k >= log_Y_line[j]-log_Y_line[k] - float(tol))
        
    m.setObjective(obj,gp.GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()

    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        beta = np.zeros(len(beta_0))
        beta[[0,1,6]] = variables[:3]
        beta[[2,3,4,5,7,8,9]] = variables[3:]
        Z = m.ObjVal
        rho,mse,mae,LL,LL_0,P = compute_metrics(Y_line,X_line,beta)
    except:
        beta = beta_0
        Z = 0
        rho,mse,mae,LL,LL_0,P = compute_metrics(Y_line,X_line,beta)
    return beta,Z,rho,mse,mae,LL,LL_0,P





# An example of N=568,K=1,J=2
# X = scipy.io.loadmat(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\LogitMLE\X.mat")['X']
# Y = scipy.io.loadmat(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\5.Group-level modeling\LogitMLE\Y.mat")['Y']
# tol = 0.001

# beta_dict,LL_dict,P,Hessian_dict,Jacobian_dict,model_summary = logitMLE(Y, X, tol)

# beta_dict,LL_dict,DIST_dict,Objective_dict,P = logitMLE_DIST(Y, X, 0.001, np.array([0,0]), lambda_=0.90)


# A real example
# with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\X_all.pickle", 'rb') as handle:
#     X = pickle.load(handle)
# with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\Y_all.pickle", 'rb') as handle:
#     Y = pickle.load(handle)
# with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\num_all.pickle", 'rb') as handle:
#     num = pickle.load(handle)
# with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\id_all.pickle", 'rb') as handle:
#     group_id = pickle.load(handle)





#TTTTTTest
# Y_line = Y[15,:]
# X_line = X[15,:,:]

# beta_0 = np.array([0,0,0,0,0,0,0,0,0,0])
# beta_name = ['auto_tt','transit_ivt','transit_at','transit_et','transit_nt','non_vehicle_tt',
#               'cost','constant_auto','constant_transit','constant_non_vehicle']

# one_choice = X_line.T
# data = pd.DataFrame(one_choice,columns=beta_name)
# data['alternative'] = pd.Series(['Auto','Transit','On_demand','Biking','Walking','Carpool'])
# data['chosen'] = Y_line


# beta,Z,rho,mse,mae,LL,LL_0,P = group_level_IO(Y_line,X_line,beta_0,tol=1,lb=-50,ub=50)
# beta




# lst = np.concatenate([np.linspace(0.01,1,99),np.linspace(1,100,99)])
# heat_map = np.zeros((len(lst),len(lst)))
# for i,v1 in enumerate(lst):
#     for j,v2 in enumerate(lst):
#         heat_map[i,j] = np.log(v1)-np.log(v2)



# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# fig, ax = plt.subplots()
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# im = ax.imshow(heat_map)
# plt.colorbar(im, cax=cax)



# np.quantile(heat_map.reshape(39204,),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
