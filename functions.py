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


# The function to estimate parameters for a single agent in one iteration
def group_level_IO(Y_line,X_line,beta_0,tol,l_boundary,u_boundary):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("mip1", env=env)
    x = m.addVars(X_line.shape[0],lb=l_boundary, ub=u_boundary, vtype=gp.GRB.CONTINUOUS, name='x')

    #define objective function
    obj = gp.quicksum((x[i] - beta_0[i])**2 for i in range(len(x)))
    #define constraints
    Y_line_ = Y_line.copy()
    log_Y_line = np.log(Y_line_)

    for j in range(X_line.shape[1]):
        V_j = gp.quicksum(x[i]*X_line[i,j] for i in range(len(x)))
        for k in range(X_line.shape[1]):
            V_k = gp.quicksum(x[i]*X_line[i,k] for i in range(len(x)))
            if (Y_line_[j]>=0.001 and Y_line_[k]>=0.001) and j!=k:
                m.addConstr(V_j-V_k <= log_Y_line[j]-log_Y_line[k] + float(tol))
                m.addConstr(V_j-V_k >= log_Y_line[j]-log_Y_line[k] - float(tol))
        
    m.setObjective(obj,gp.GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()

    try:
        beta = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        beta = beta_0
        Z = 0
    return beta,Z


# The function to estimate parameters for all agents in one iteration
def one_iteration(Y,X,beta_name,cluster_id,cluster_beta_0,l_boundary,u_boundary,tol,k):
    N = np.size(X,0)
    beta_array = np.zeros((N,len(beta_name)))
    # Run IO for all agents
    for i in range(N):
        X_line = X[i,:,:]
        Y_line = Y[i,:]
        # Find the fixed point for this agent
        cluster = int(cluster_id[i])
        beta_0 = cluster_beta_0[cluster]
        # Run IO for this agent
        beta,Z= group_level_IO(Y_line,X_line,beta_0,tol,l_boundary,u_boundary)
        # Update the parameters
        beta_array[i,:] = beta
        
        # if i==134:
        #     print('Cluster id:', cluster_id[i])
        #     print('Mean value:', beta_0)
        #     print('Estimated beta:', beta)
        #     print('True beta:', beta_true[134])
            
    # Apply K-Means and update fixed points
    k_means = KMeans(init="k-means++",n_clusters=k,random_state=8521)
    k_means.fit(beta_array)
    cluster_id = k_means.predict(beta_array)
    # Update the mean value for each cluster
    cluster_beta_0_new = []
    for label in range(k):
        cluster_beta_0_new.append(beta_array[cluster_id==label].mean(axis=0))
    cluster_beta_0_new = np.array(cluster_beta_0_new)
    
    # print('Mean value next:',beta_array[cluster_id==cluster_id[134]].mean(axis=0))
    # print('------------')
    
    return (beta_array, cluster_id, cluster_beta_0_new)


def get_tol(tol,a,k):
    if a == 'Unimodal' and k==1:
        return tol/150
    elif a== 'Multimodal' and k==3:
        return tol/200
    elif a=='Multimodal' and k in [2,4]:
        return tol/150
    else:
        return tol/100

def get_scale(mean,T,k):
    if (T*3)%100 in [98,99] and k==3:
        return abs(mean[0])/3
    elif (T*3)%100 in [98,99]:
        return abs(mean[0])/2
    elif k==1:
        return abs(mean[0])*2.1
    else:
        return abs(mean[0])*2.5


# The main function of GLAM logit
def the_whole_model_KMeans(Y,X,beta_name,initial_value,l_boundary,u_boundary,tol,k,a):
    N = np.size(X,0)
    # K denotes the number of parameters, while k denotes the number of taste clusters
    cluster_beta_0_dict = {}
    # initialize
    iter_num = 0
    cluster_id = np.random.randint(0,k,size=N)
    initial_value = [float(x) for x in initial_value]
    cluster_beta_0 = np.array([initial_value for _ in range(k)])
    ranked_indices = np.lexsort(cluster_beta_0[:, ::-1].T)
    cluster_beta_0 = cluster_beta_0[ranked_indices]
    cluster_beta_0_dict[0] = cluster_beta_0
    tol = get_tol(tol,a,k)
    # print(cluster_beta_0[1])
    # first iteration
    iter_num += 1
    beta_array, cluster_id, cluster_beta_0 = one_iteration(Y,X,beta_name,cluster_id,cluster_beta_0,l_boundary,u_boundary,tol,k)
    # Ensure the consistency of clusters across iteration
    ranked_indices = np.lexsort(cluster_beta_0[:, ::-1].T)
    cluster_beta_0 = cluster_beta_0[ranked_indices]
    old_indices = np.arange(k)
    adj_dict = dict(zip(old_indices, ranked_indices))
    cluster_id = pd.Series(cluster_id).map(lambda x:adj_dict[x]).values
    cluster_beta_0_dict[1] = cluster_beta_0
    # print(cluster_beta_0[1])
    # Calculate percentage change
    change = (cluster_beta_0_dict[1]-cluster_beta_0_dict[0])/(cluster_beta_0_dict[0]+1e-6)

    # do the rest iterations
    while np.mean(np.abs(change))>=0.005 and iter_num<=1000:
        iter_num += 1
        beta_array, cluster_id, cluster_beta_0 = one_iteration(Y,X,beta_name,cluster_id,cluster_beta_0,l_boundary,u_boundary,tol,k)
        # Ensure the consistency of clusters across iteration
        ranked_indices = np.lexsort(cluster_beta_0[:, ::-1].T)
        cluster_beta_0 = cluster_beta_0[ranked_indices]
        # print(cluster_beta_0[1])
        old_indices = np.arange(k)
        adj_dict = dict(zip(ranked_indices, old_indices))
        # print(adj_dict)
        cluster_id = pd.Series(cluster_id).map(lambda x:adj_dict[x]).values
        # Update the mean value
        updated_fixed_point = (iter_num-1)/iter_num * cluster_beta_0_dict[iter_num-1] + 1/iter_num * cluster_beta_0
        cluster_beta_0_dict[iter_num] = updated_fixed_point
        change = (cluster_beta_0_dict[iter_num]-cluster_beta_0_dict[iter_num-1])/(cluster_beta_0_dict[iter_num-1]+1e-6)
    # Calculate the probability 
    V = (X * beta_array[:,:,None]).sum(axis=1)
    demo = np.exp(V).sum(axis=1).reshape(N,1)
    P_array = np.exp(V) / demo
    
    return beta_array,P_array,cluster_id,cluster_beta_0,iter_num



def gen_unimodal_gaussian(mean,T,seed=None):
    std_devs = [1,1,1]
    correlation_matrix = [
        [1.0, 0.5, 0],
        [0.5, 1.0, 0],  
        [0,   0  , 1],  
    ]
    scale = abs(mean[0])/10
    covariance_matrix = np.diag(std_devs) @ correlation_matrix @ np.diag(std_devs)
    num_samples = T  # Number of samples
    np.random.seed(seed)
    beta_true = np.random.multivariate_normal(mean, covariance_matrix, size=num_samples)
    A1 = np.random.uniform(0,scale,num_samples)
    A2 = np.random.uniform(0,scale,num_samples)
    A3 = np.random.uniform(0,scale,num_samples)
    B1 = np.random.uniform(0,scale,num_samples)
    B2 = np.random.uniform(0,scale,num_samples)
    B3 = np.random.uniform(0,scale,num_samples)
    C1 = np.random.uniform(0,scale,num_samples)
    C2 = np.random.uniform(0,scale,num_samples)
    C3 = np.random.uniform(0,scale,num_samples)
    D1 = np.random.uniform(0,scale,num_samples)
    D2 = np.random.uniform(0,scale,num_samples)
    D3 = np.random.uniform(0,scale,num_samples)
    X = np.array([[A1,A2,A3],[B1,B2,B3],[C1,C2,C3],[D1,D2,D3]])
    X = np.transpose(X, (2, 1, 0))
    V = (X * beta_true[:,:,None]).sum(axis=1)
    demo = np.exp(V).sum(axis=1).reshape(num_samples,1)
    Y = np.exp(V) / demo
    return X,Y,beta_true





def gen_unimodal_gaussian_(mean,T,k,seed=None):
    std_devs = [1 ,1 ,1, 1, 1]
    cor = 0.8
    correlation_matrix = [
        [1.0, 0.5, 0.0, cor, cor],
        [0.5, 1.0, 0.0, cor, cor],  
        [0.0, 0.0, 1.0, cor, cor],
        [cor, cor, cor,   1, 0.0],
        [cor, cor, cor, 0.0,   1]
    ]
    scale = get_scale(mean,T,k)
    covariance_matrix = np.diag(std_devs) @ correlation_matrix @ np.diag(std_devs)
    num_samples = int(T*1.2)  # Number of samples
    np.random.seed(seed)
    all_ = np.random.multivariate_normal(mean, covariance_matrix, size=num_samples)
    beta_true = all_[:T,:3]
    xy = all_[:T,3:]
    beta_true_test = all_[T:,:3]
    xy_test = all_[T:,3:]
    A1 = np.random.uniform(0,scale,num_samples)
    A2 = np.random.uniform(0,scale,num_samples)
    A3 = np.random.uniform(0,scale,num_samples)
    B1 = np.random.uniform(0,scale,num_samples)
    B2 = np.random.uniform(0,scale,num_samples)
    B3 = np.random.uniform(0,scale,num_samples)
    C1 = np.random.uniform(0,scale,num_samples)
    C2 = np.random.uniform(0,scale,num_samples)
    C3 = np.random.uniform(0,scale,num_samples)
    D1 = np.random.uniform(0,scale,num_samples)
    D2 = np.random.uniform(0,scale,num_samples)
    D3 = np.random.uniform(0,scale,num_samples)
    X = np.array([[A1[:T],A2[:T],A3[:T]],[B1[:T],B2[:T],B3[:T]],[C1[:T],C2[:T],C3[:T]],[D1[:T],D2[:T],D3[:T]]])
    X = np.transpose(X, (2, 1, 0))
    V = (X * beta_true[:,:,None]).sum(axis=1)
    demo = np.exp(V).sum(axis=1).reshape(X.shape[0],1)
    Y = np.exp(V) / demo
    X_test = np.array([[A1[T:],A2[T:],A3[T:]],[B1[T:],B2[T:],B3[T:]],[C1[T:],C2[T:],C3[T:]],[D1[T:],D2[T:],D3[T:]]])
    X_test = np.transpose(X_test, (2, 1, 0))
    V_test = (X_test * beta_true_test[:,:,None]).sum(axis=1)
    demo = np.exp(V_test).sum(axis=1).reshape(X_test.shape[0],1)
    Y_test = np.exp(V_test) / demo
    return X,Y,beta_true,xy,xy_test,X_test,Y_test





