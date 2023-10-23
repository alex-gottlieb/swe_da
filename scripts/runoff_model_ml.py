#!/usr/bin/env python
# coding: utf-8

# In[67]:


import os 
import xarray as xr
from itertools import product
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import geopandas as gpd
import sys


# In[68]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


# In[69]:


tmean = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','tmean_mon.nc'))
ppt = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','ppt_mon.nc'))
swe = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs.nc'))


# In[70]:


tmean_prods = list(tmean['product'].values)
ppt_prods = list(ppt['product'].values)
swe_prods = list(swe['product'].values)
combos = list(product(swe_prods,ppt_prods,tmean_prods))

grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other','grdc_basins'))
area_da = xr.Dataset.from_dataframe(grdc_basins.set_index("MRBID")[['AREA_CALC']])
area_da = area_da.rename({"MRBID":"basin","AREA_CALC":"area"})


ind = int(sys.argv[1])
combo = combos[ind]
combo_str = '_'.join(combo)

hist_dir = os.path.join(project_dir,'data','basin_scale','hist_recons')


noacc_dir = os.path.join(project_dir,'data','basin_scale','noacc_recons')
noacc_files = [os.path.join(noacc_dir,f) for f in os.listdir(noacc_dir)]
noacc_files.sort()

hist_file = os.path.join(hist_dir,f'{combo_str}.csv')
noacc_files = [f for f in noacc_files if combo_str in f]


hist_swe = xr.Dataset.from_dataframe(pd.read_csv(hist_file).set_index(['basin','time'])[['swe_pred']])
hist_swe['time'] = hist_swe['time'].astype('datetime64[ns]')
hist_swe['swe'] = 1e6*(hist_swe['swe_pred']/area_da['area'])


noacc_swe = [xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['basin','time']))[['swe_noacc']].assign_coords(combo=f.split("/")[-1].split(".")[0]) for f in noacc_files]
noacc_swe = xr.concat(noacc_swe,dim='combo')
noacc_swe['time'] = noacc_swe['time'].astype('datetime64[ns]')
noacc_swe['swe'] = 1e6*(noacc_swe['swe_noacc']/area_da['area'])



hist_swe = hist_swe.where(swe['swe'].min(['product','time'])>0,drop=True)
noacc_swe = noacc_swe.where(swe['swe'].min(['product','time'])>0,drop=True)


q = xr.open_dataset(os.path.join(project_dir,'gridded_model','data','runoff_model','amjj_q_basin.nc')).resample(time='AS-OCT').mean().sel(time=slice("1980","2019"))
q['q'] = 1e6*(q['q']/area_da['area'])

ppt['ppt'] = 1e6*(ppt['ppt']/area_da['area'])


swe_prod,ppt_prod,tmean_prod=combo
merged = q
for m in [11,12,1,2,3,4,5,6,7]:
    merged[f'tmean_{m}'] = tmean['tmean'].sel(product=tmean_prod).sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
    merged[f'ppt_{m}'] = ppt['ppt'].sel(product=ppt_prod).sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()
merged = merged.drop("product")
merged = xr.merge([merged,hist_swe['swe']],join='inner')
df = merged.to_dataframe()
rf  = RandomForestRegressor(random_state=42,n_jobs=-1).fit(df.dropna().iloc[:,1:],df.dropna().iloc[:,0],)

pred_df = df.dropna(subset=['swe'])[['q']]
pred_df = pred_df.rename(columns={"q":"q_obs"})
pred_df['q_pred'] = rf.predict(df.dropna(subset=['swe']).iloc[:,1:])
pred_df.index.names = ['basin','time']
pred_ds = xr.Dataset.from_dataframe(pred_df)

noacc_q = []
for c in noacc_swe['combo'].values:
    merged['swe'] = noacc_swe['swe'].sel(combo=c)
    df = merged.to_dataframe()
    pred_noacc = pd.DataFrame(rf.predict(df.dropna().iloc[:,1:-1]),index=df.dropna().index,columns=['q_noacc'])
    pred_noacc['combo'] = c
    pred_noacc.index.names = ['basin','time']
    noacc_q.append(pred_noacc)
noacc_q = pd.concat(noacc_q)
noacc_q = noacc_q.reset_index().set_index(['basin','time','combo'])
noacc_q = xr.Dataset.from_dataframe(noacc_q)
noacc_q['combo'] = noacc_q['combo'].astype(str)
noacc_q.to_netcdf(os.path.join(project_dir,'data','basin_scale','runoff_recons_noacc',f'{combo_str}.nc'))

fut_swe = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','swe_fut.nc'))
fut_swe = fut_swe.where(fut_swe['combo'].str.startswith(combo_str),drop=True)
fut_swe['swe_fut_clim']=1e6*fut_swe['swe_fut_clim']/area_da['area']
hist_clim = merged.sel(basin=fut_swe['basin']).mean('time')
hist_clim_df = hist_clim.to_dataframe()
hist_clim_df['q_clim'] = rf.predict(hist_clim_df.iloc[:,1:-1])
hist_clim_df.index.names = ['basin']
fut_clim = xr.merge([hist_clim.drop("swe"),fut_swe[['swe_fut_clim']]],join='inner')
fut_clim_df = fut_clim.to_dataframe()
fut_clim_df['q_fut'] = rf.predict(fut_clim_df.iloc[:,1:])
fut_clim_df.index.names = ['basin','combo']
q_df = hist_clim_df[['q_clim']].merge(fut_clim_df[['q_fut']],left_index=True,right_index=True)

q_ds = xr.Dataset.from_dataframe(q_df)
pred_ds['q_fut'] = q_ds['q_fut']
pred_ds.to_netcdf(os.path.join(project_dir,'data','basin_scale','runoff_fut',f"{combo_str}.nc"))