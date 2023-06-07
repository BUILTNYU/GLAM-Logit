# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:37:08 2022

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
import time
from math import radians, cos, sin, asin, sqrt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import gurobipy as gp
from gurobipy import GRB
from matplotlib_scalebar.scalebar import ScaleBar

path = 'k2'
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\beta_array.pickle'%path, 'rb') as handle:
    beta_array = pickle.load(handle)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\P_array.pickle'%path, 'rb') as handle:
    P_array = pickle.load(handle)
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\cluster_id_new.pickle'%path, 'rb') as handle:
    cluster_id_new = pickle.load(handle)  
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\cluster_beta_0_dict.pickle'%path, 'rb') as handle:
    cluster_beta_0_dict = pickle.load(handle) 
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\LL_dict.pickle'%path, 'rb') as handle:
    LL_dict = pickle.load(handle)    
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\DIST_dict.pickle'%path, 'rb') as handle:
    DIST_dict = pickle.load(handle)    
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\MSE_dict.pickle'%path, 'rb') as handle:
    MSE_dict = pickle.load(handle)  
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\MAE_dict.pickle'%path, 'rb') as handle:
    MAE_dict = pickle.load(handle)     
with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\model_results\%s\feasible_array.pickle'%path, 'rb') as handle:
    feasible_array = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\id_all.pickle", 'rb') as handle:
    group_id = pickle.load(handle)
    
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\X_all.pickle", 'rb') as handle:
    X = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\Y_all.pickle", 'rb') as handle:
    Y = pickle.load(handle)
with open(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\num_all.pickle", 'rb') as handle:
    num = pickle.load(handle)




###################################
###Create a new mobility service###
###################################

mode = ['Private_auto','Public_transit','On_demand','Biking','Walking','Carpool']
beta_name = ['auto_tt','transit_ivt','transit_at','transit_et','transit_nt','non_vehicle_tt',
              'cost','constant_auto','constant_transit','constant_non_vehicle']

# Create a (120740,10) shaped array for the new mobility service
new_service = np.zeros((np.size(X,0),np.size(X,1))) 
new_service[:,0] = X[:,0,2]+5/60 # travel time is similar to on_demand travel time plus 5 min waiting time
new_service[:,1:6] = 0
new_service[:,6] = X[:,6,2]/2 # travel cost is half of on_demand travel
new_service[:,7] = 1 # an auto service
new_service[:,8:] = 0

# Append the new mobility service to the original choice set
X_new = X.copy()
X_new = np.append(X_new,new_service[:,:,None],axis=2)
X_new.shape

# Calculate the change of social welfare and mode share
V = (beta_array[:,:,None]*X).sum(axis=1)
sum_e_V = (np.exp(V)).sum(axis=1)
social_welfare = np.log(sum_e_V/np.size(X,2))
V_new = (beta_array[:,:,None]*X_new).sum(axis=1)
sum_e_V_new = (np.exp(V_new)).sum(axis=1)
social_welfare_new = np.log(sum_e_V_new/np.size(X_new,2))
change_of_welfare = social_welfare_new - social_welfare
print(change_of_welfare.sum())

P_new = np.exp(V_new)/(np.exp(V_new)).sum(axis=1)[:,None]

###########################################
###Combine population segments and links###
###########################################
all_agents = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\shapefile\all_agents.shp")
all_agents.rename(columns={'length':'trip_length'},inplace=True)
all_agents['trip_length'] = all_agents['trip_length']*np.sqrt(2)
all_agents['welfare_old'] = social_welfare
all_agents['welfare_new'] = social_welfare_new
all_agents['change_of_welfare'] = change_of_welfare
all_agents['new_mode_share'] = P_new[:,-1]
all_agents['new_mobility_fee'] = new_service[:,6]
all_agents.rename(columns={'origin_bgr':'origin_bgrp','destinatio':'destination_bgrp'},inplace=True)

disadvantage = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\3.Data Quality\Disadvantaged communities\disad_areas.shp")
all_agents['is_disadvantage'] = all_agents['origin_bgrp'].isin(disadvantage['GEOID'].unique())

all_agents['sum_welfare_old'] = all_agents['welfare_old']*all_agents['Trip_num']
all_agents['sum_welfare_new'] = all_agents['welfare_new']*all_agents['Trip_num']
all_agents['new_mobility_demand'] = all_agents['new_mode_share']*all_agents['Trip_num']
all_agents['sum_new_fee'] = all_agents['new_mobility_fee']*all_agents['Trip_num']
all_brgp_links = all_agents.groupby(['origin_bgrp','destination_bgrp']).agg({'trip_length':'first','geometry':'first','is_disadvantage':'first','sum_welfare_old':'sum','sum_welfare_new':'sum','new_mobility_demand':'sum','sum_new_fee':'sum','Trip_num':'sum'}).reset_index()

all_brgp_links['welfare_old'] = all_brgp_links['sum_welfare_old']/all_brgp_links['Trip_num']
all_brgp_links['welfare_new'] = all_brgp_links['sum_welfare_new']/all_brgp_links['Trip_num']
all_brgp_links['new_mobility_fee'] = all_brgp_links['sum_new_fee']/all_brgp_links['Trip_num']
all_brgp_links['change_of_welfare'] = all_brgp_links['welfare_new']-all_brgp_links['welfare_old']
all_brgp_links = all_brgp_links[all_brgp_links['change_of_welfare']<5]
all_brgp_links['change_of_welfare'].hist(bins=100)

all_brgp_links['O_County'] = all_brgp_links['origin_bgrp'].map(lambda x: x[:5])
all_brgp_links['D_County'] = all_brgp_links['destination_bgrp'].map(lambda x: x[:5])
all_brgp_links = all_brgp_links[all_brgp_links['O_County']==all_brgp_links['D_County']]
all_brgp_links = all_brgp_links.reset_index()
all_brgp_links.drop('index',axis=1,inplace=True)
all_brgp_links = gpd.GeoDataFrame(all_brgp_links,geometry=gpd.GeoSeries(all_brgp_links['geometry']),crs='4326')
NY_blockgroup = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\3.Data Quality\NY_blockgroup\tl_2019_36_bg.shp")[['GEOID','geometry']]
NY_county = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\NYS_Civil_Boundaries.shp\Counties.shp")


###################################
###Zone-based bi-direction model###
###################################
class SocialWelfareOptimizor:
    def __init__(
        self,
        all_agents: gpd.GeoDataFrame,
        NY_blockgroup: gpd.GeoDataFrame,
        NY_county: gpd.GeoDataFrame,
        max_operating_zones: int,
        max_fleet_size_total: int,
        max_fleet_size_county: int,
        min_fleet_size_county: int,
        max_VKT: int,
        max_trip: int,
        model: str,
    ):
        self.all_agents = all_agents
        self.NY_blockgroup = NY_blockgroup
        self.NY_county = NY_county
        self.max_operating_zones = max_operating_zones
        self.max_fleet_size_total = max_fleet_size_total
        self.max_fleet_size_county = max_fleet_size_county
        self.min_fleet_size_county = min_fleet_size_county
        self.max_VKT = max_VKT
        self.max_trip = max_trip
        self.model = model
        
    def solve_model(self):
        county_list = self.all_agents['O_County'].unique()
        m = gp.Model()
        # Decision variables
        zone = m.addVars(len(county_list),vtype=GRB.BINARY,name='zone')
        link = m.addVars(len(self.all_agents),vtype=GRB.BINARY,name='link')
        fleet = m.addVars(len(self.all_agents),vtype=GRB.INTEGER,name='fleet')
        
        # Constraints
        m.addConstr(gp.quicksum(zone[i] for i in range(len(county_list))) <= self.max_operating_zones)
        m.addConstr(gp.quicksum(fleet[i] for i in range(len(self.all_agents))) <= self.max_fleet_size_total)
        
        for i in range(len(county_list)):
            index_list = self.all_agents[self.all_agents['O_County']==county_list[i]].index
            m.addConstr(gp.quicksum(link[j] for j in index_list) <= zone[i] * 1e10)
            # to ensure link diversity
            m.addConstr(gp.quicksum(link[j] for j in index_list) >= zone[i] * 10)
            m.addConstr(gp.quicksum(fleet[j] for j in index_list) <= zone[i] * self.max_fleet_size_county)
            m.addConstr(gp.quicksum(fleet[j] for j in index_list) >= zone[i] * self.min_fleet_size_county)
        for i in range(len(self.all_agents)):
            m.addConstr(fleet[i]<=link[i] * 1e10)
            m.addConstr(fleet[i]*self.max_VKT>=link[i]*self.all_agents['new_mobility_demand'][i]*self.all_agents['trip_length'][i])
            m.addConstr(fleet[i]*self.max_trip>=link[i]*self.all_agents['new_mobility_demand'][i])
        for i in range(len(self.all_agents)):
            condition1 = self.all_agents['origin_bgrp']==self.all_agents['destination_bgrp'].iloc[i]
            condition2 = self.all_agents['destination_bgrp']==self.all_agents['origin_bgrp'].iloc[i]
            pair_index = self.all_agents[condition1 & condition2].index
            if len(pair_index)==1:
                m.addConstr(fleet[i]==fleet[np.squeeze(pair_index)])
        # Objective function
        if self.model=='Model0':
            m.setObjective(gp.quicksum(link[i]*self.all_agents['new_mobility_demand'][i]*self.all_agents['new_mobility_fee'][i] for i in range(len(self.all_agents))),GRB.MAXIMIZE)
        elif self.model=='Model1':
            m.setObjective(gp.quicksum(link[i]*self.all_agents['Trip_num'][i]*self.all_agents['change_of_welfare'][i] for i in range(len(self.all_agents))),GRB.MAXIMIZE)
        elif self.model=='Model2':
            dis = gp.quicksum(link[i]*self.all_agents['Trip_num'][i]*self.all_agents['change_of_welfare'][i] for i in range(len(self.all_agents)) if self.all_agents['is_disadvantage'][i]==True)
            other = gp.quicksum(link[i]*self.all_agents['Trip_num'][i]*self.all_agents['change_of_welfare'][i] for i in range(len(self.all_agents)) if self.all_agents['is_disadvantage'][i]==False)
            m.setObjective(dis-other,GRB.MAXIMIZE)
        else:
            return 'unknown model'
        # Solve the model
        m.update()
        m.Params.LogToConsole = 0
        m.optimize()
        self.m = m

    def retrieve_model(self):
        return self.m

    def get_results(self):
        results = self.all_agents.copy()
        try:
            Z = self.m.ObjVal
            decision_variable = np.array(self.m.getAttr('X', self.m.getVars()))
        except:
            return 'infeasible','infeasible','infeasible'
        
        county_list = results['O_County'].unique()
        served_county = decision_variable[:len(county_list)]
        link = decision_variable[len(county_list):len(county_list)+len(results)]
        fleet = decision_variable[len(county_list)+len(results):]
        served_county_list = county_list[served_county==1]
        results['link'] = link
        results['fleet_size'] = fleet*2
        return results, Z, served_county_list
        
    def plot_results(self,save=True,name='aaa'):  
        NY_brgp = self.NY_blockgroup.copy()
        NY_C = self.NY_county.copy()
        results, Z, served_county_list = self.get_results()
        NY_brgp = NY_brgp.to_crs('2263')
        NY_C = NY_C.to_crs('2263')
        results = results.to_crs('2263')
        
        fig,ax = plt.subplots(figsize=(8,6))
        NY_C[NY_C['FIPS_CODE'].isin(served_county_list)].plot(facecolor='grey',ax=ax)
        NY_brgp.plot(facecolor='lightgrey',edgecolor='white',alpha=0.6,linewidth=0.1,ax=ax)
        if results['fleet_size'].max()>0:
            results[(results['fleet_size']>0)&(results['fleet_size']<=50)].plot(linewidth=0.2,ax=ax,color='cornflowerblue',label='0-50 vehicles')
        if results['fleet_size'].max()>50:
            results[(results['fleet_size']>0)&(results['fleet_size']>50)&(results['fleet_size']<=100)].plot(linewidth=0.3,ax=ax,color='orange',label='50-100 vehicles')
        if results['fleet_size'].max()>100:
            results[(results['fleet_size']>0)&(results['fleet_size']>100)&(results['fleet_size']<=200)].plot(linewidth=0.5,ax=ax,color='orangered',label='100-200 vehicles')
        if results['fleet_size'].max()>200:
            results[(results['fleet_size']>0)&(results['fleet_size']>200)].plot(linewidth=0.7,ax=ax,color='darkred',label='more than 200 vehicles')
        # ax.add_artist(ScaleBar(1))
        ax.axis('off')
        plt.legend(loc='lower left')
        plt.title('Optimal Solution: (%i zones, %i vehicles)'%(self.max_operating_zones,self.max_fleet_size_total))
        if save==True:
            plt.savefig('C:/Users/MSI-PC/Desktop/%s.jpg'%name,
                dpi=600,
                bbox_inches = 'tight',
                facecolor = 'w',
                edgecolor = 'w')
            results[results['fleet_size']>0].to_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\shapefile\OR model results\%s_link.shp"%name)
            NY_C[NY_C['FIPS_CODE'].isin(served_county_list)].to_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\shapefile\OR model results\%s_county.shp"%name)

    def calculate_metrics(self,base=True):
        results, Z, served_county_list = self.get_results()
        results['welfare_final'] = results['link'].map(lambda x: 0 if x==1 else 1)*results['welfare_old'] + results['link']*results['welfare_new']
        results2 = results[results['O_County'].isin(served_county_list)]
        metrics = {}
        # service information
        metrics['num_zone'] = len(served_county_list)
        metrics['num_link'] = int(results['link'].sum())
        metrics['num_fleet'] = int(results['fleet_size'].sum()/2)
        metrics['num_trips'] = int((results['link']*results['new_mobility_demand']).sum())
        metrics['avg_VKT'] = (results['link']*results['new_mobility_demand']*results['trip_length']).sum()/metrics['num_fleet']
        metrics['revenue'] = (results['link']*results['new_mobility_demand']*results['new_mobility_fee']).sum()
        # Average welfare
        results2 = results
        SW = (results2['welfare_final']*results2['Trip_num']).sum()/results2['Trip_num'].sum()
        SW_0 = (results2['welfare_old']*results2['Trip_num']).sum()/results2['Trip_num'].sum()
        metrics['Average_welfare_2'] = SW
        metrics['Pct_average_welfare_2'] = (SW-SW_0)/SW_0*100
        # Welfare range
        R =  results2['welfare_final'].max()-results2['welfare_final'].min()
        R_0 = results2['welfare_old'].max()-results2['welfare_old'].min()
        metrics['Welfare_range'] = R
        metrics['Pct_welfare_range'] = (R-R_0)/R_0*100
        # Welfare mean deviation
        MD = np.abs(results2['welfare_final']-results2['welfare_final'].mean()).sum()/len(results2)
        MD_0 = np.abs(results2['welfare_old']-results2['welfare_old'].mean()).sum()/len(results2)
        metrics['Mean_deviation'] = MD
        metrics['Pct_mean_deviation'] = (MD-MD_0)/MD_0*100
        # Welfare variation
        Var = results2['welfare_final'].var()
        Var_0 = results2['welfare_old'].var()
        metrics['Variation'] = Var
        metrics['Pct_variation'] = (Var-Var_0)/Var_0*100
        # Gini coefficient
        def gini(x):
            total = 0
            for i, xi in enumerate(x[:-1], 1):
                total += np.sum(np.abs(xi - x[i:]))
            return total / (len(x)**2 * np.mean(x))
        Gini = gini(results2['welfare_final'].values)
        Gini_0 = gini(results2['welfare_old'].values)
        metrics['Gini'] = Gini
        metrics['Pct_gini'] = (Gini-Gini_0)/Gini_0*100
        # Pairwise differences
        PD = results2[results2['is_disadvantage']==True]['welfare_final'].mean()-results2[results2['is_disadvantage']==False]['welfare_final'].mean()
        PD_0 = results2[results2['is_disadvantage']==True]['welfare_old'].mean()-results2[results2['is_disadvantage']==False]['welfare_old'].mean()
        metrics['PD'] = PD
        metrics['Pct_pd'] = (PD-PD_0)/PD_0*100
        if base==True:
            metrics['Average_welfare_base_2'] = SW_0
            metrics['Welfarw_range_base'] = R_0
            metrics['Mean_deviation_base'] = MD_0
            metrics['Variation_base'] = Var_0
            metrics['Gini_base'] = Gini_0
            metrics['PD_base'] = PD_0
        return metrics
        
        


# full run
max_operating_zones = 20
max_fleet_size_total = 20000
max_fleet_size_county = max_fleet_size_total/max_operating_zones*2
min_fleet_size_county = max_fleet_size_total/max_operating_zones/2
max_VKT = 200
max_trip = 10

inputs = all_brgp_links[all_brgp_links['trip_length']>=1].reset_index()

Model0 = SocialWelfareOptimizor(inputs,NY_blockgroup,NY_county,max_operating_zones,
                                max_fleet_size_total,max_fleet_size_county,min_fleet_size_county,
                                max_VKT,max_trip,model='Model0')
Model1 = SocialWelfareOptimizor(inputs,NY_blockgroup,NY_county,max_operating_zones,
                                max_fleet_size_total,max_fleet_size_county,min_fleet_size_county,
                                max_VKT,max_trip,model='Model1')
Model2 = SocialWelfareOptimizor(inputs,NY_blockgroup,NY_county,max_operating_zones,
                                max_fleet_size_total,max_fleet_size_county,min_fleet_size_county,
                                max_VKT,max_trip,model='Model2')


start_time = time.time()
Model0.solve_model()
Model0.plot_results(save=True,name='Model0_%i_%i'%(max_operating_zones,max_fleet_size_total))
metrics_0 = Model0.calculate_metrics()
end_time = time.time()
print('model running time: %is'%int(end_time-start_time))

start_time = time.time()
Model1.solve_model()
Model1.plot_results(save=True,name='Model1_%i_%i'%(max_operating_zones,max_fleet_size_total))
metrics_1 = Model1.calculate_metrics()
end_time = time.time()
print('model running time: %is'%int(end_time-start_time))

start_time = time.time()
Model2.solve_model()
Model2.plot_results(save=True,name='Model2_%i_%i'%(max_operating_zones,max_fleet_size_total))
metrics_2 = Model2.calculate_metrics()
end_time = time.time()
print('model running time: %is'%int(end_time-start_time))




with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\metrics_of_OR\20_20000_metrics_0.pickle', 'wb') as handle:
    pickle.dump(metrics_0, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\metrics_of_OR\20_20000_metrics_1.pickle', 'wb') as handle:
    pickle.dump(metrics_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\metrics_of_OR\20_20000_metrics_2.pickle', 'wb') as handle:
    pickle.dump(metrics_2, handle, protocol=pickle.HIGHEST_PROTOCOL)






















