# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:32:08 2022

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
from math import radians, cos, sin, asin, sqrt
from Group_level_IO import logitMLE_DIST,logitMLE

#########################################
##Load data from BQ and get coordinates##
#########################################
filepath = r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\Data From BQ\LowIncome_OD_20.csv"
data_type = {'origin_bgrp':str, 'destination_bgrp':str}
OD_data = pd.read_csv(filepath,encoding='ANSI',dtype=data_type)
OD_data_2 = OD_data[OD_data['origin_bgrp']!=OD_data['destination_bgrp']]
# OD_data_2 = OD_data[OD_data['Trip_num']>=10]
NY_bgrp = gpd.read_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\7.NY mode choice model_2\NY region\NY_blockgroup_centroid.shp")[['GEOID','geometry']]
NY_bgrp['lng'] = NY_bgrp['geometry'].x
NY_bgrp['lat'] = NY_bgrp['geometry'].y
OD_data_2 = pd.merge(OD_data_2,NY_bgrp,left_on='origin_bgrp',right_on='GEOID')
OD_data_2 = pd.merge(OD_data_2,NY_bgrp,left_on='destination_bgrp',right_on='GEOID')
OD_data_2.rename(columns={'lng_x':'lng_o','lat_x':'lat_o','lng_y':'lng_d','lat_y':'lat_d'},inplace=True)

def geodistance(record):
    lng1,lat1,lng2,lat2 = map(radians,[float(record['lng_o']), float(record['lat_o']), float(record['lng_d']), float(record['lat_d'])])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance,4)
    return distance

OD_data_2['distance_meters'] = OD_data_2.apply(geodistance,axis=1)

OD_data_2 = OD_data_2[['origin_bgrp','destination_bgrp','lng_o','lat_o','lng_d','lat_d',
                       'Trip_num','distance_meters',
                       'Pro_PRIVATE_AUTO','Pro_PUBLIC_TRANSIT','Pro_ON_DEMAND_AUTO',
                       'Pro_BIKING','Pro_WALKING','Pro_CARPOOL',
                       'Dur_PRIVATE_AUTO','Dur_PUBLIC_TRANSIT','Dur_ON_DEMAND_AURTO',
                       'Dur_BIKING','Dur_WALKING','Dur_CARPOOL',
                       'Access_time','Egress_time','In_vehicle_time','Num_transfer',
                       'Cost_PRIVATE_AUTO','Cost_PUBLIC_TRANSIT','Cost_ON_DEMAND_AURTO',
                       'Cost_BIKING','Cost_WALKING','Cost_CARPOOL']]

OD_data_2['Dur_PRIVATE_AUTO'] = OD_data_2['Dur_PRIVATE_AUTO']/60
OD_data_2['Dur_PUBLIC_TRANSIT'] = OD_data_2['Dur_PUBLIC_TRANSIT']/60
OD_data_2['Access_time'] = OD_data_2['Access_time']/60
OD_data_2['Egress_time'] = OD_data_2['Egress_time']/60
OD_data_2['In_vehicle_time'] = OD_data_2['In_vehicle_time']/60
OD_data_2['Dur_ON_DEMAND_AURTO'] = OD_data_2['Dur_ON_DEMAND_AURTO']/60
OD_data_2['Dur_BIKING'] = OD_data_2['Dur_BIKING']/60
OD_data_2['Dur_WALKING'] = OD_data_2['Dur_WALKING']/60
OD_data_2['Dur_CARPOOL'] = OD_data_2['Dur_CARPOOL']/60

OD_check = OD_data_2[(OD_data_2['Pro_PRIVATE_AUTO']>0)&(OD_data_2['Pro_PUBLIC_TRANSIT']>0)&(OD_data_2['Pro_ON_DEMAND_AUTO']>0)&(OD_data_2['Pro_BIKING']>0)&(OD_data_2['Pro_WALKING']>0)&(OD_data_2['Pro_CARPOOL']>0)]

# OD_check['In_vehicle_time'].mean()/OD_check['Dur_PUBLIC_TRANSIT'].mean()

###############################################
##Fill none value & revise wrong value per OD##
###############################################
dur_columns = ['Dur_PRIVATE_AUTO','Dur_PUBLIC_TRANSIT','Dur_ON_DEMAND_AURTO','Dur_BIKING','Dur_WALKING','Dur_CARPOOL']

def private_auto_time(record):
    if record['Dur_PRIVATE_AUTO']>0:
        return record['Dur_PRIVATE_AUTO']
    else:
        if record['Dur_ON_DEMAND_AURTO']>0:
            factor = OD_check['Dur_PRIVATE_AUTO'].mean()/OD_check['Dur_ON_DEMAND_AURTO'].mean()
            return record['Dur_ON_DEMAND_AURTO']*factor*0.9
        elif record['Dur_CARPOOL']>0:
            factor = OD_check['Dur_PRIVATE_AUTO'].mean()/OD_check['Dur_CARPOOL'].mean()
            return record['Dur_CARPOOL']*factor*0.8
        else:
            return record[dur_columns].max()

def transit_time(record):
    if record['Dur_PUBLIC_TRANSIT']>0:
        return record['Dur_PUBLIC_TRANSIT']
    else:
        factor = OD_check['Dur_PUBLIC_TRANSIT'].mean()/OD_check['Dur_PRIVATE_AUTO'].mean()
        return record['Dur_PRIVATE_AUTO']*factor/1.5

def carpool_time(record):
    if record['Dur_CARPOOL']>0:
        return record['Dur_CARPOOL']
    else:
        factor = OD_check['Dur_CARPOOL'].mean()/OD_check['Dur_PRIVATE_AUTO'].mean()
        return record['Dur_PRIVATE_AUTO']*factor

def ondemand_time(record):
    if record['Dur_ON_DEMAND_AURTO']>0:
        return record['Dur_ON_DEMAND_AURTO']
    else:
        factor = OD_check['Dur_ON_DEMAND_AURTO'].mean()/OD_check['Dur_PRIVATE_AUTO'].mean()
        return record['Dur_PRIVATE_AUTO']*factor

def walking_time(record):
    if record['Dur_WALKING']>0:
        return record['Dur_WALKING']
    else:
        factor = OD_check['Dur_WALKING'].mean()/OD_check['Dur_PRIVATE_AUTO'].mean()
        return record['Dur_PRIVATE_AUTO']*factor

def biking_time(record):
    if record['Dur_BIKING']>0:
        return record['Dur_BIKING']
    else:
        factor = OD_check['Dur_BIKING'].mean()/OD_check['Dur_PRIVATE_AUTO'].mean()
        return record['Dur_PRIVATE_AUTO']*factor


OD_data_2['Dur_PRIVATE_AUTO'] = OD_data_2.apply(private_auto_time,axis=1)
OD_data_2['Dur_PUBLIC_TRANSIT'] = OD_data_2.apply(transit_time,axis=1)*0.75
OD_data_2['Dur_ON_DEMAND_AURTO'] = OD_data_2.apply(ondemand_time,axis=1)*1.2
OD_data_2['Dur_BIKING'] = OD_data_2.apply(biking_time,axis=1)
OD_data_2['Dur_WALKING'] = OD_data_2.apply(walking_time,axis=1)
OD_data_2['Dur_CARPOOL'] = OD_data_2.apply(carpool_time,axis=1)*1.5

def invehicle_time(record):
    if record['In_vehicle_time']>0:
        return record['In_vehicle_time']
    else:
        factor = OD_check['In_vehicle_time'].mean()/OD_check['Dur_PUBLIC_TRANSIT'].mean()
        return record['Dur_PUBLIC_TRANSIT']*factor

def access_time(record):
    if record['In_vehicle_time']>0:
        return record['Access_time']
    else:
        factor = OD_check['Access_time'].mean()/OD_check['Dur_PUBLIC_TRANSIT'].mean()
        return record['Dur_PUBLIC_TRANSIT']*factor

def egress_time(record):
    if record['In_vehicle_time']>0:
        return record['Egress_time']
    else:
        factor = OD_check['Egress_time'].mean()/OD_check['Dur_PUBLIC_TRANSIT'].mean()
        return record['Dur_PUBLIC_TRANSIT']*factor

def num_transfer(record):
    if record['Num_transfer']>=0:
        return record['Num_transfer']
    else:
        return np.random.rand()

OD_data_2['Access_time'] = OD_data_2.apply(access_time,axis=1)
OD_data_2['Egress_time'] = OD_data_2.apply(egress_time,axis=1)
OD_data_2['In_vehicle_time'] = OD_data_2.apply(invehicle_time,axis=1)
OD_data_2['Num_transfer'] = OD_data_2.apply(num_transfer,axis=1)


cost_columns = ['Cost_PRIVATE_AUTO','Cost_PUBLIC_TRANSIT','Cost_ON_DEMAND_AURTO','Cost_BIKING','Cost_WALKING','Cost_CARPOOL']

def driving_cost(record):
    if record['Cost_PRIVATE_AUTO']>0:
        return record['Cost_PRIVATE_AUTO']
    else:
        return OD_check['Cost_PRIVATE_AUTO'].mean()

def transit_cost(record):
    if filepath.split('\\')[-1].split('_')[0]=='Senior':
        factor=0.5
    else:
        factor=1
    if record['origin_bgrp'][:5] in ('36005','36047','36061','36081','36085') or record['destination_bgrp'][:5] in ('36005','36047','36061','36081','36085'):
        return 2.75*factor
    else:
        return 2.75/2*factor

def carpool_cost(record):
    if record['Cost_CARPOOL']>0:
        return record['Cost_CARPOOL']
    elif record['Cost_PRIVATE_AUTO']>0:
        return record['Cost_PRIVATE_AUTO']/2
    else:
        return record[cost_columns].max()

def ondemand_cost(record):
    if record['Cost_ON_DEMAND_AURTO']>0:
        return record['Cost_ON_DEMAND_AURTO']
    else:
        return record[cost_columns].max()

OD_data_2['Cost_PRIVATE_AUTO'] = OD_data_2.apply(driving_cost,axis=1)
OD_data_2['Cost_PUBLIC_TRANSIT'] = OD_data_2.apply(transit_cost,axis=1)
OD_data_2['Cost_ON_DEMAND_AURTO'] = OD_data_2.apply(ondemand_cost,axis=1)
OD_data_2['Cost_BIKING'] = 0
OD_data_2['Cost_WALKING'] = 0
# OD_data_2['Cost_CARPOOL'] = OD_data_2.apply(carpool_cost,axis=1)
OD_data_2['Cost_CARPOOL'] = (OD_data_2['Cost_PRIVATE_AUTO']+OD_data_2['Cost_ON_DEMAND_AURTO'])/2

OD_data_2.to_csv('NY_Processed_OD_LowIncome.csv',index=False)




###########################
##Group-level IO building##
###########################
data_type = {'origin_bgrp':str, 'destination_bgrp':str}
dur_columns = ['Dur_PRIVATE_AUTO','Dur_PUBLIC_TRANSIT','Dur_ON_DEMAND_AURTO',
               'Dur_BIKING','Dur_WALKING','Dur_CARPOOL']
transit_columns = ['Access_time','Egress_time','In_vehicle_time','Num_transfer']
cost_columns = ['Cost_PRIVATE_AUTO','Cost_PUBLIC_TRANSIT','Cost_ON_DEMAND_AURTO',
                'Cost_BIKING','Cost_WALKING','Cost_CARPOOL']
Pro_column = ['Pro_PRIVATE_AUTO','Pro_PUBLIC_TRANSIT','Pro_ON_DEMAND_AUTO',
              'Pro_BIKING','Pro_WALKING','Pro_CARPOOL']

NotLowIncome = pd.read_csv(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\NY_Processed_OD_NotLowIncome.csv",dtype=data_type)
NotLowIncome['Population'] = 'NotLowIncome'

LowIncome = pd.read_csv(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\NY_Processed_OD_LowIncome.csv",dtype=data_type)
LowIncome['Population'] = 'LowIncome'

Senior = pd.read_csv(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\NY_Processed_OD_Senior.csv",dtype=data_type)
Senior['Population'] = 'Senior'

Student = pd.read_csv(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\NY_Processed_OD_Student.csv",dtype=data_type)
Student['Population'] = 'Student'

data = pd.concat([NotLowIncome,LowIncome,Senior,Student])



# get geodata
from shapely.geometry import Point, LineString
for_geo = data[['origin_bgrp', 'destination_bgrp', 'lng_o', 'lat_o', 'lng_d', 'lat_d', 'Trip_num','Population']].reset_index()
geometry_O = [Point(xy) for xy in zip(for_geo.lng_o, for_geo.lat_o)]
geometry_D = [Point(xy) for xy in zip(for_geo.lng_d, for_geo.lat_d)]
OD_lst = []
for i in range(len(for_geo)):
    OD = LineString([geometry_O[i],geometry_D[i]])
    OD_lst.append(OD)
    
g = gpd.GeoSeries(OD_lst)
OD_link = gpd.GeoDataFrame(for_geo.iloc[:,1:],geometry=g,crs="EPSG:4326")
# OD_link.to_file(r"C:\Users\MSI-PC\Desktop\Social Equity Project with Replica\9.NY mode choice model final version\shapefile\all_agents.shp")


def cal_group_id(record):
    return (record['origin_bgrp']+'_'+record['destination_bgrp']+'_'+record['Population'])

data['group_id'] = data.apply(cal_group_id,axis=1)


auto = np.array([1,0,1,0,0,1])
transit = np.array([0,1,0,0,0,0])
biking_walking = np.array([0,0,0,1,1,0])

Y_lst = []
X_lst = []
for i in range(len(data)):
    a_line = data.iloc[i]
    dur_auto = np.array([a_line['Dur_PRIVATE_AUTO'],0,a_line['Dur_ON_DEMAND_AURTO'],0,0,a_line['Dur_CARPOOL']])
    dur_invehicle = np.array([0,a_line['In_vehicle_time'],0,0,0,0])
    dur_access = np.array([0,a_line['Access_time'],0,0,0,0])
    dur_egress = np.array([0,a_line['Egress_time'],0,0,0,0])
    num_transfer = np.array([0,a_line['Num_transfer'],0,0,0,0])
    dur_biking_walking = np.array([0,0,0,a_line['Dur_BIKING'],a_line['Dur_WALKING'],0])
    cost = a_line[cost_columns].values
    Y_lst.append(a_line[Pro_column].values)
    X_lst.append(np.row_stack([dur_auto,dur_invehicle,dur_access,dur_egress,num_transfer,dur_biking_walking,
                               cost,auto,transit,biking_walking])[None,:,:])

Y = np.row_stack(Y_lst).astype(np.float)
X = np.row_stack(X_lst).astype(np.float)

beta_0 = np.array([0,0,0,0,0,0,0,0,0,0])
beta_name = ['auto_tt','transit_ivt','transit_at','transit_et','transit_nt','non_vehicle_tt',
             'cost','constant_auto','constant_transit','constant_non_vehicle']


# bb = data.groupby('group_id').agg({'Dur_PRIVATE_AUTO':'count','Trip_num':'sum'})

# #test_whole
# beta_dict,LL_dict,DIST_dict,Objective_dict,P = logitMLE_DIST(Y, X, 0.001, beta_0, beta_name, lambda_=10)


#test_a_line
# lst = [Y[0,:][None,:]]
# Y_ = np.row_stack(lst*2)

# lst = [X[0,:,:][None,:,:]]
# X_ = np.row_stack(lst*2)

# beta_dict,LL_dict,DIST_dict,Objective_dict,P = logitMLE_DIST(Y_, X_, 0.001, beta_0, beta_name, lambda_=10)


#Output——np.array
num = np.array(data['Trip_num'].values)
group_id = np.array(data['group_id'].values)

with open('X_all.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Y_all.pickle', 'wb') as handle:
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('num_all.pickle', 'wb') as handle:
    pickle.dump(num, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('id_all.pickle', 'wb') as handle:
    pickle.dump(group_id, handle, protocol=pickle.HIGHEST_PROTOCOL)








