#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import xarray as xr
from rasterio import features
from affine import Affine
import numpy as np
import geopandas as gpd
import sys
import warnings
from itertools import product
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import xesmf as xe
warnings.filterwarnings("ignore")


# In[ ]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','swe_da')
data_dir = os.path.join(project_dir,'data','regrid_0.5deg')
swe_dir = os.path.join(data_dir,'swe')
ppt_dir = os.path.join(data_dir,'ppt')
tmean_dir = os.path.join(data_dir,'tmean')

swe_prods = os.listdir(swe_dir)
swe_prods.sort()
ppt_prods = os.listdir(ppt_dir)
ppt_prods.sort()
tmean_prods = os.listdir(tmean_dir)
tmean_prods.sort()

# all unique SWE-P-T data product combinations
combos = list(product(swe_prods,ppt_prods,tmean_prods))

# choose which combo to run (argument from shell script)
ind = int(sys.argv[1])
combo = combos[ind]
swe_prod,ppt_prod,tmean_prod=combo
combo_str = '_'.join(combo).replace(".nc","")


swe_mask =  xr.open_dataset(os.path.join(project_data_dir,'regrid_0.5deg','swe_mask.nc')) # keep cells where >1/2 of all product years have non-zero March SWE
gl_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_0.5deg','gl_mask.nc')) # mask out Greenland

# load data
swe = xr.open_dataset(os.path.join(swe_dir,swe_prod)).sel(time=slice("1980","2020"))
ppt = xr.open_dataset(os.path.join(ppt_dir,ppt_prod)).sel(time=slice("1980-11-01","2020-03-31"))
tmean = xr.open_dataset(os.path.join(tmean_dir,tmean_prod)).sel(time=slice("1980-11-01","2020-03-31"))

# get monthly T and P from November through March
merged = swe.resample(time='AS-OCT').mean().sel(time=slice("1980","2019"))
for m in [11,12,1,2,3]:
    merged[f'tmean_{m}'] = tmean['tmean'].sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
    merged[f'ppt_{m}'] = ppt['ppt'].sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()


# apply masks
merged = merged.where((gl_mask['gl_mask']==1)&(swe_mask==1))

# convert to dataframe for fitting model
df = merged.to_dataframe()
df = df.dropna(axis=0,how='any',subset=list(df.columns)[1:])

basins = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','basins.nc'))
basin_df = basins.to_dataframe()
area = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','area.nc'))
area_df = area.to_dataframe()

# fit Random Forest model
rf  = RandomForestRegressor(random_state=42,n_jobs=-1).fit(df.dropna().iloc[:,1:],df.dropna().iloc[:,0],)

# get predictions from model
pred_df = df[['swe']]
pred_df = pred_df.rename(columns={"swe":"swe_obs"})
pred_df['swe_pred'] = rf.predict(df.iloc[:,1:])

# convert back to xarray
pred_ds = xr.Dataset.from_dataframe(pred_df[['swe_pred']])
pred_ds = pred_ds.sortby("lon")
pred_ds = pred_ds.reindex_like(swe,'nearest')

# aggregate to basin-scale
basin_pred_df = pred_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
basin_pred_df['swe_obs'] = basin_pred_df['swe_obs']*basin_pred_df['area']
basin_pred_df['swe_pred'] = basin_pred_df['swe_pred']*basin_pred_df['area']
basin_pred_df = basin_pred_df.groupby(['basin',basin_pred_df.index.get_level_values("time")])[['swe_obs','swe_pred',]].sum()/1e12
basin_pred_df.to_csv(os.path.join(project_dir,'data','basin_scale','hist_recons',f'{combo_str}.csv'))


# forced response from cmip6 models
ppt_fr_dir = os.path.join(project_dir,'data','ppt_mon_cmip6','forced_resp_30y')
ppt_fr_mods = os.listdir(ppt_fr_dir)

tmean_fr_dir = os.path.join(project_dir,'data','tmean_mon_cmip6','forced_resp_30y')
tmean_fr_mods = os.listdir(tmean_fr_dir)

shared_mods = list(set(ppt_fr_mods)&(set(tmean_fr_mods)))
shared_mods.sort()

from time import time
for mod in shared_mods:
    if os.path.exists(os.path.join(project_dir,'data','basin_scale','noacc_recons',"_".join([combo_str,mod[:-3]])+".csv")):
        continue
    t0 = time()
    
    # regrid  forced responses to half-degree
    weight_fn = os.path.join(project_dir,'ml_model','data','cmip_regridder_weights',f'{mod.split("_")[0]}.nc')
    with xr.open_dataset(os.path.join(tmean_fr_dir,mod)) as t_fr:
        if not os.path.exists(weight_fn):
            regridder = xe.Regridder(t_fr,tmean,'bilinear')
            regridder.to_netcdf(weight_fn)
        regridder = xe.Regridder(t_fr,tmean,'bilinear',weights=weight_fn)
        t_fr = regridder(t_fr)
        t_fr['time'] = t_fr['time'].astype("datetime64[ns]")
                             
        # remove forced temperature increase
        tmean_noacc = tmean['tmean'].resample(time='1M').mean()-t_fr['delta_t']
        tmean_noacc.name = 'tmean'

    with xr.open_dataset(os.path.join(ppt_fr_dir,mod)) as p_fr:
        regridder = xe.Regridder(p_fr,ppt,'bilinear',weights=weight_fn)
        p_fr = regridder(p_fr)
        p_fr['time'] = p_fr['time'].astype("datetime64[ns]")
        
        # remove forced precip increase
        ppt_noacc = ppt['ppt'].resample(time='1M').mean()/p_fr['delta_p']
        ppt_noacc.name = 'ppt'

    # create counterfactual datasets 
    # counterfactual T & P
    noacc = []
    for m in [11,12,1,2,3]:
        _tmean = tmean_noacc.sel(time=tmean_noacc['time'][tmean_noacc['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc.append(_tmean)

        _ppt = ppt_noacc.sel(time=ppt_noacc['time'][ppt_noacc['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc.append(_ppt)
    noacc = xr.merge(noacc)

    # only counterfactual P
    noacc_p = []
    for m in [11,12,1,2,3]:
        _tmean = tmean['tmean'].sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc_p.append(_tmean)

        _ppt = ppt_noacc.sel(time=ppt_noacc['time'][ppt_noacc['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc_p.append(_ppt)
    noacc_p = xr.merge(noacc_p)

    # only counterfactual T
    noacc_t = []
    for m in [11,12,1,2,3]:
        _tmean = tmean_noacc.sel(time=tmean_noacc['time'][tmean_noacc['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc_t.append(_tmean)

        _ppt = ppt['ppt'].sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc_t.append(_ppt)
    noacc_t = xr.merge(noacc_t)

    # mask and convert to dataframe
    noacc = noacc.where((gl_mask['gl_mask']==1)&(swe_mask==1))
    noacc_df = noacc.to_dataframe()
    noacc_df = noacc_df.replace([np.inf, -np.inf], np.nan)
    noacc_df = noacc_df.dropna(axis=0,how='any',subset=list(noacc_df.columns)[1:])

    noacc_p = noacc_p.where((gl_mask['gl_mask']==1)&(swe_mask==1))
    noacc_p_df = noacc_p.to_dataframe()
    noacc_p_df = noacc_p_df.replace([np.inf, -np.inf], np.nan)
    noacc_p_df = noacc_p_df.dropna(axis=0,how='any',subset=list(noacc_p_df.columns)[1:])

    noacc_t = noacc_t.where((gl_mask['gl_mask']==1)&(swe_mask==1))
    noacc_t_df = noacc_t.to_dataframe()
    noacc_t_df = noacc_t_df.replace([np.inf, -np.inf], np.nan)
    noacc_t_df = noacc_t_df.dropna(axis=0,how='any',subset=list(noacc_t_df.columns)[1:])

    # predict counterfactuals using model trained on historical data
    noacc_df['swe_pred_noacc'] = rf.predict(noacc_df[list(df.columns)[1:]])
    noacc_p_df['swe_pred_noacc_p'] = rf.predict(noacc_p_df[list(df.columns)[1:]])
    noacc_t_df['swe_pred_noacc_t'] = rf.predict(noacc_t_df[list(df.columns)[1:]])

    # convert to basin-scale snow mass
    basin_noacc_df = noacc_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
    basin_noacc_df['swe_noacc'] = basin_noacc_df['swe_pred_noacc']*basin_noacc_df['area']
    basin_noacc_df = basin_noacc_df.groupby(['basin',basin_noacc_df.index.get_level_values("time")])[['swe_noacc']].sum()/1e12

    basin_noacc_p_df = noacc_p_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
    basin_noacc_p_df['swe_noacc_p'] = basin_noacc_p_df['swe_pred_noacc_p']*basin_noacc_p_df['area']
    basin_noacc_p_df = basin_noacc_p_df.groupby(['basin',basin_noacc_p_df.index.get_level_values("time")])[['swe_noacc_p']].sum()/1e12

    basin_noacc_t_df = noacc_t_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
    basin_noacc_t_df['swe_noacc_t'] = basin_noacc_t_df['swe_pred_noacc_t']*basin_noacc_t_df['area']
    basin_noacc_t_df = basin_noacc_t_df.groupby(['basin',basin_noacc_t_df.index.get_level_values("time")])[['swe_noacc_t']].sum()/1e12

    basin_noacc_pred_df = basin_noacc_df.merge(basin_noacc_p_df,left_index=True,right_index=True).merge(basin_noacc_t_df,left_index=True,right_index=True)
    basin_noacc_pred_df.to_csv(os.path.join(project_dir,'data','basin_scale','noacc_recons',f'{combo_str}_{mod[:-3]}.csv'))
    t1 = time()
    print(t1-t0)
                           