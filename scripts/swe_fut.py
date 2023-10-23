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
from datetime import datetime
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


# In[ ]:


combos = list(product(swe_prods,ppt_prods,tmean_prods))
ind = int(sys.argv[1])
combo = combos[ind]
swe_prod,ppt_prod,tmean_prod=combo
combo_str = '_'.join(combo).replace(".nc","")


# In[ ]:


mask = xr.open_dataset(os.path.join(data_dir,'gl_mask.nc'))


# In[ ]:


swe = xr.open_dataset(os.path.join(swe_dir,swe_prod)).sel(time=slice("1980","2020"))
ppt = xr.open_dataset(os.path.join(ppt_dir,ppt_prod)).sel(time=slice("1980-11-01","2020-03-31"))
tmean = xr.open_dataset(os.path.join(tmean_dir,tmean_prod)).sel(time=slice("1980-11-01","2020-03-31"))

# In[ ]:


merged = swe.resample(time='AS-OCT').mean().sel(time=slice("1980","2019"))
for m in [11,12,1,2,3]:
    merged[f'tmean_{m}'] = tmean['tmean'].sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
    merged[f'ppt_{m}'] = ppt['ppt'].sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()

# In[ ]:


merged = merged.where(mask['gl_mask']>0.9)
merged['swe'] = merged['swe'].where(merged['swe'].median("time")>0)
df = merged.to_dataframe()
df = df.dropna(axis=0,how='any',subset=list(df.columns)[1:])

basins = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','basins.nc'))
basin_df = basins.to_dataframe()
area = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','area.nc'))
area_df = area.to_dataframe()

# fit model to historical data
rf  = RandomForestRegressor(random_state=42,n_jobs=-1).fit(df.dropna().iloc[:,1:],df.dropna().iloc[:,0],)
pred_df = df[['swe']]
pred_df = pred_df.rename(columns={"swe":"swe_obs"})
pred_df['swe_pred'] = rf.predict(df.iloc[:,1:])

# aggregate to basin-scale
basin_pred_df = pred_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
basin_pred_df['swe_obs'] = basin_pred_df['swe_obs']*basin_pred_df['area']
basin_pred_df['swe_pred'] = basin_pred_df['swe_pred']*basin_pred_df['area']
basin_pred_df = basin_pred_df.groupby(['basin',basin_pred_df.index.get_level_values("time")])[['swe_obs','swe_pred']].sum()/1e12


ppt_hist_dir = os.path.join(project_dir,'data','ppt_mon_cmip6','historical')
ppt_hist_mods = os.listdir(ppt_hist_dir)

ppt_fut_dir = os.path.join(project_dir,'data','ppt_mon_cmip6','ssp245')
ppt_fut_mods = os.listdir(ppt_fut_dir)

tmean_hist_dir = os.path.join(project_dir,'data','tmean_mon_cmip6','historical')
tmean_hist_mods = os.listdir(tmean_hist_dir)

tmean_fut_dir = os.path.join(project_dir,'data','tmean_mon_cmip6','ssp245')
tmean_fut_mods = os.listdir(tmean_fut_dir)

shared_mods = list(set(ppt_hist_mods)&set(ppt_fut_mods)&set(tmean_hist_mods)&set(tmean_fut_mods))
shared_mods.sort()

obs_tmean_clim = tmean.groupby("time.month").mean()
obs_ppt_clim = ppt.groupby("time.month").mean()
    
for mod in shared_mods:
    if os.path.exists(os.path.join(project_dir,'ml_model','data','recons_fut',f"{combo_str}_{mod}.csv")):
        continue
    try:
        weight_fn = os.path.join(project_dir,'ml_model','data','cmip_regridder_weights',f'{mod.split("_")[0]}.nc')
        with xr.open_dataset(os.path.join(tmean_hist_dir,mod)) as t_hist:
            with xr.open_dataset(os.path.join(tmean_fut_dir,mod)) as t_fut:
                t_hist_clim = t_hist.sel(time=slice("1981","2020")).groupby("time.month").mean()
                t_fut_clim = t_fut.sel(time=slice("2070","2099")).groupby("time.month").mean()
                delta_t = t_fut_clim-t_hist_clim
                if not os.path.exists(weight_fn):
                    regridder = xe.Regridder(delta_t,tmean,'bilinear')
                    regridder.to_netcdf(weight_fn)
                else:
                    regridder = xe.Regridder(delta_t,tmean,'bilinear',weights=weight_fn)
                delta_t = regridder(delta_t)

        with xr.open_dataset(os.path.join(ppt_hist_dir,mod)) as p_hist:
            with xr.open_dataset(os.path.join(ppt_fut_dir,mod)) as p_fut:
                p_hist_clim = p_hist.sel(time=slice("1981","2020")).groupby("time.month").mean()
                p_fut_clim = p_fut.sel(time=slice("2070","2099")).groupby("time.month").mean()
                delta_p = p_fut_clim/p_hist_clim
                regridder = xe.Regridder(delta_p,ppt,'bilinear',weights=weight_fn)
                delta_p = regridder(delta_p)


        fut_clim = []
        for m in [11,12,1,2,3]:
            _tmean = obs_tmean_clim.sel(month=m)+delta_t.sel(month=m)
            _tmean = _tmean.rename({"tmean":f'tmean_{m}'})
            fut_clim.append(_tmean.drop(["month","height"]))

            _ppt = obs_ppt_clim.sel(month=m)*delta_p.sel(month=m)
            _ppt = _ppt.rename({'ppt':f'ppt_{m}'})
            fut_clim.append(_ppt.drop("month"))
        fut_clim = xr.merge(fut_clim)
        fut_clim = fut_clim.assign_coords(time=[datetime(2099,10,1)])
        fut_clim = fut_clim.where(mask['gl_mask']>0.9)
        fut_clim_df = fut_clim.to_dataframe()
        fut_clim_df = fut_clim_df.replace([np.inf, -np.inf], np.nan)
        fut_clim_df = fut_clim_df.dropna(axis=0,how='any',subset=list(fut_clim_df.columns)[1:])
        fut_clim_df['swe_pred_fut_clim'] = rf.predict(fut_clim_df[list(df.columns)[1:]])
        basin_fut_clim_df = fut_clim_df.merge(area_df,left_index=True,right_index=True).merge(basin_df,left_index=True,right_index=True)
        basin_fut_clim_df['swe_fut_clim'] = basin_fut_clim_df['swe_pred_fut_clim']*basin_fut_clim_df['area']
        basin_fut_clim_df = basin_fut_clim_df.groupby(['basin',basin_fut_clim_df.index.get_level_values("time")])[['swe_fut_clim']].sum()/1e12
        swe_clim = basin_pred_df.groupby("basin").mean()[['swe_pred']]
        swe_clim = swe_clim.rename(columns={"swe_pred":"swe_clim"})
        basin_fut_clim_df = basin_fut_clim_df.merge(swe_clim,left_index=True,right_index=True)
        basin_fut_clim_df.to_csv(os.path.join(project_dir,'ml_model','data','recons_fut',f"{combo_str}_{mod}.csv"))
    except:
         continue