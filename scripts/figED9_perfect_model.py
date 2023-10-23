#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
import pymannkendall as mk
warnings.filterwarnings("ignore")


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
data_dir = os.path.join(project_dir,'data','regrid_2deg')
swe_dir = os.path.join(data_dir,'snw_hist')
ppt_dir = os.path.join(data_dir,'ppt_hist')
tmean_dir = os.path.join(data_dir,'tmean_hist')
tmean_hn_dir = os.path.join(data_dir,'tmean_hist-nat')
ppt_hn_dir = os.path.join(data_dir,'ppt_hist-nat')

swe_mods = os.listdir(swe_dir)
swe_mods.sort()
ppt_mods = os.listdir(ppt_dir)
ppt_mods.sort()
ppt_hn_mods = os.listdir(ppt_hn_dir)
ppt_hn_mods.sort()
tmean_mods = os.listdir(tmean_dir)
tmean_mods.sort()
tmean_hn_mods = os.listdir(tmean_hn_dir)
tmean_hn_mods.sort()

shared_mods = list(set(swe_mods)&set(ppt_mods)&set(tmean_mods)&set(ppt_hn_mods)&set(tmean_hn_mods))
shared_mods.sort()


# In[7]:


gl_mask = xr.open_dataset(os.path.join(data_dir,'gl_mask.nc')) # Greenland mask
swe_mask = xr.open_dataset(os.path.join(data_dir,'swe_mask.nc')) # SWE mask


# In[ ]:


from time import time
for mod in shared_prods:
    if os.path.exists(os.path.join(project_dir,'data','regrid_2deg','recons_cmip6',mod)):
        continue
    t0 = time()
    swe = xr.open_dataset(os.path.join(swe_dir,mod))
    ppt = xr.open_dataset(os.path.join(ppt_dir,mod))
    tmean = xr.open_dataset(os.path.join(tmean_dir,mod))

    if mod == 'GFDL-CM4_r1i1p1f1.nc':
        swe['time'] = swe['time'].astype('datetime64[ns]')
        tmean['time'] = tmean['time'].astype('datetime64[ns]')
        ppt['time'] = ppt['time'].astype('datetime64[ns]')

    merged = swe.resample(time='AS-OCT').mean().sel(time=slice("1980","2019"))
    for m in [11,12,1,2,3]:
        merged[f'tmean_{m}'] = tmean['tmean'].sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
        merged[f'ppt_{m}'] = ppt['ppt'].sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()
    merged = merged.where((gl_mask['gl_mask']==1)&(swe_mask['swe']==1))
    merged['snw'] = merged['snw'].clip(max=800)
    df = merged.to_dataframe()
    df = df.dropna(axis=0,how='any',subset=list(df.columns)[1:]).drop(columns=['model','height'])
    rf  = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1,verbose=0).fit(X=df.dropna().iloc[:,1:],y=df.dropna().iloc[:,0])

    pred_df = df[['snw']]
    pred_df = pred_df.rename(columns={"snw":"snw_orig"})
    pred_df['snw_pred'] = rf.predict(df.iloc[:,1:])
    pred_ds = xr.Dataset.from_dataframe(pred_df[['snw_orig','snw_pred']])
    pred_ds = pred_ds.sortby("lon")
    pred_ds = pred_ds.reindex_like(swe,'nearest')

    tmean_hn = xr.open_dataset(os.path.join(tmean_hn_dir,mod))
    ppt_hn = xr.open_dataset(os.path.join(ppt_hn_dir,mod))

    if mod == 'GFDL-CM4_r1i1p1f1.nc':
        tmean_hn['time'] = tmean_hn['time'].astype('datetime64[ns]')
        ppt_hn['time'] = ppt_hn['time'].astype('datetime64[ns]')
        
    noacc = []
    for m in [11,12,1,2,3]:
        _tmean = tmean_hn['tmean'].sel(time=tmean_hn['time'][tmean_hn['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc.append(_tmean.drop("height"))

        _ppt = ppt_hn['ppt'].sel(time=ppt_hn['time'][ppt_hn['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc.append(_ppt)
    noacc = xr.merge(noacc)

    noacc_t = []
    for m in [11,12,1,2,3]:
        _tmean = tmean_hn['tmean'].sel(time=tmean_hn['time'][tmean_hn['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc_t.append(_tmean.drop("height"))

        _ppt = ppt['ppt'].sel(time=ppt['time'][ppt['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc_t.append(_ppt)
    noacc_t = xr.merge(noacc_t)

    noacc_p = []
    for m in [11,12,1,2,3]:
        _tmean = tmean['tmean'].sel(time=tmean['time'][tmean['time.month']==m]).resample(time='AS-OCT').mean()
        _tmean.name = f'tmean_{m}'
        noacc_p.append(_tmean)

        _ppt = ppt_hn['ppt'].sel(time=ppt_hn['time'][ppt_hn['time.month']==m]).resample(time='AS-OCT').mean()
        _ppt.name = f'ppt_{m}'
        noacc_p.append(_ppt)
    noacc_p = xr.merge(noacc_p)

    noacc = noacc.where(mask['gl_mask']>0.9)
    noacc_df = noacc.to_dataframe()
    noacc_df = noacc_df.replace([np.inf, -np.inf], np.nan)
    noacc_df = noacc_df.dropna(axis=0,how='any',subset=list(noacc_df.columns)[1:])

    noacc_p = noacc_p.where(mask['gl_mask']>0.9)
    noacc_p_df = noacc_p.to_dataframe()
    noacc_p_df = noacc_p_df.replace([np.inf, -np.inf], np.nan)
    noacc_p_df = noacc_p_df.dropna(axis=0,how='any',subset=list(noacc_p_df.columns)[1:])

    noacc_t = noacc_t.where(mask['gl_mask']>0.9)
    noacc_t_df = noacc_t.to_dataframe()
    noacc_t_df = noacc_t_df.replace([np.inf, -np.inf], np.nan)
    noacc_t_df = noacc_t_df.dropna(axis=0,how='any',subset=list(noacc_t_df.columns)[1:])

    noacc_df['swe_pred_noacc'] = rf.predict(noacc_df[list(df.columns)[1:]])
    noacc_p_df['swe_pred_noacc_p'] = rf.predict(noacc_p_df[list(df.columns)[1:]])
    noacc_t_df['swe_pred_noacc_t'] = rf.predict(noacc_t_df[list(df.columns)[1:]])

    noacc_ds = xr.Dataset.from_dataframe(noacc_df[['swe_pred_noacc']])
    noacc_ds = noacc_ds.sortby("lon")
    noacc_ds = noacc_ds.reindex_like(swe,'nearest')

    noacc_p_ds = xr.Dataset.from_dataframe(noacc_p_df[['swe_pred_noacc_p']])
    noacc_p_ds = noacc_p_ds.sortby("lon")
    noacc_p_ds = noacc_p_ds.reindex_like(swe,'nearest')

    noacc_t_ds = xr.Dataset.from_dataframe(noacc_t_df[['swe_pred_noacc_t']])
    noacc_t_ds = noacc_t_ds.sortby("lon")
    noacc_t_ds = noacc_t_ds.reindex_like(swe,'nearest')

    combined = xr.merge([pred_ds,noacc_ds,noacc_p_ds,noacc_t_ds])
    combined.to_netcdf(os.path.join(project_dir,'ml_model','data','recons_cmip6',mod))
    t1 = time()
    print(mod,t1-t0)


# In[8]:


recon_dir = os.path.join(data_dir,'recons_cmip6')
recon_mods = os.listdir(recon_dir)
recon_mods.sort()

swe_hn_dir = os.path.join(data_dir,'snw_hist-nat')
swe_hn_mods = os.listdir(swe_hn_dir)
swe_hn_mods.sort()

shared = list(set(swe_prods)&set(recon_mods)&set(swe_hn_mods))
shared.sort()


# In[10]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope
    except:
        return np.nan


# In[13]:


def fix_time(ds):
    return [datetime(y,3,16) for y in ds['time.year'].values]

fr_trends = []
fr_ens = []
hist_ens = []
for mod in shared:
    if ('r1i1p1f1' not in mod) & ('CNRM-CM6-1' not in mod):
        continue
    hist = xr.open_dataset(os.path.join(swe_dir,mod))
    hist['time'] = fix_time(hist)
    histnat = xr.open_dataset(os.path.join(swe_hn_dir,mod))
    recon = xr.open_dataset(os.path.join(recon_dir,mod))

    hist_trend = xr.apply_ufunc(theil_sen,hist['snw'],input_core_dims=[['time']],vectorize=True)
    histnat_trend = xr.apply_ufunc(theil_sen,histnat['snw'],input_core_dims=[['time']],vectorize=True)
    rec_hist_trend = xr.apply_ufunc(theil_sen,recon['snw_pred'],input_core_dims=[['time']],vectorize=True)
    rec_histnat_trend = xr.apply_ufunc(theil_sen,recon['swe_pred_noacc'],input_core_dims=[['time']],vectorize=True)

    hist_trend = 1000*hist_trend/hist['snw'].mean("time")
    histnat_trend = 1000*histnat_trend/histnat['snw'].mean("time")
    rec_hist_trend = 1000*rec_hist_trend/recon['snw_pred'].mean("time")
    rec_histnat_trend = 1000*rec_histnat_trend/recon['swe_pred_noacc'].mean("time")

    orig_fr_trend = hist_trend-histnat_trend
    recon_fr_trend = rec_hist_trend-rec_histnat_trend

    orig_fr_trend.name = 'fr_orig'
    recon_fr_trend.name = 'fr_recon'
    fr_ds = xr.merge([orig_fr_trend,recon_fr_trend])
    fr_ds = fr_ds.assign_coords(model=mod.split(".")[0])
    fr_trends.append(fr_ds)
    hist = hist.assign_coords(model=mod.split(".")[0])
    hist_ens.append(hist)
    print(mod)
fr_trends = xr.concat(fr_trends,dim='model')
hist_ens = xr.concat(hist_ens,dim='model')

hist_ens = hist_ens.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))
fr_trends = fr_trends.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))


# In[16]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
import seaborn as sns
from scipy.stats import gaussian_kde,pearsonr,spearmanr
sns.set(style='ticks',font_scale=1.3)

levels = np.linspace(-25,25,11)
cols = plt.get_cmap('RdBu')(np.linspace(0,0.99,len(levels)+1))
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-25,vmax=25)

fig = plt.figure(figsize=(22,36))
gs = gridspec.GridSpec(nrows=13,ncols=3,height_ratios=12*[1]+[0.2],width_ratios=[1,1,0.4],wspace=0.01,figure=fig)
axes = []
for i,mod in enumerate(fr_trends['model'].values):
    ax1 = plt.subplot(gs[i,0],projection=ccrs.Miller())
    fr_trends['fr_orig'].sel(model=mod).plot(ax=ax1,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,add_colorbar=False)
    ax1.text(-0.03,0.5,mod.split("_")[0],ha='right',va='center',transform=ax1.transAxes,rotation=90)
    ax2 = plt.subplot(gs[i,1],projection=ccrs.Miller())
    fr_trends['fr_recon'].sel(model=mod).plot(ax=ax2,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,add_colorbar=False)
    if i == 0:
        ax1.title.set_text('A. MODEL OUTPUT SWE')
        ax2.title.set_text('B. RECONSTRUCTED SWE')
    else:
        ax1.title.set_text("")
        ax2.title.set_text("")
    axes.append(ax1)
    axes.append(ax2)
    
    ax3 = plt.subplot(gs[i,2])
    df = fr_trends.sel(model=mod).to_dataframe().drop(columns=['model']).dropna()
    x = df['fr_recon'].values
    y = df['fr_orig'].values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax3.scatter(x,y,c=z,cmap='inferno',s=50)
    ax3.set_xlim(-40,40)
    ax3.set_ylim(-40,40)
    ax3.plot([-40,40],[-40,40],color='black',linestyle='--')
    ax3.axhline(0,color='black')
    ax3.axvline(0,color='black')
    r = spearmanr(df['fr_recon'],df['fr_orig'])[0]
    ax3.text(0.05,0.9,f"r={np.round(r,2)}",ha='left',va='top',transform=ax3.transAxes)
    ax3.set_xticklabels([])
    ax3.yaxis.tick_right()
    ax3.set_ylabel("MODEL OUTPUT\n(%/DECADE)",rotation=-90,labelpad=30)
    ax3.yaxis.set_label_position("right")
    ax3.set_aspect("equal")
    if i==0:
        ax3.title.set_text("C.")
    else:
        ax3.title.set_text("")

    
    
ax1 = plt.subplot(gs[i+1,0],projection=ccrs.Miller())
fr_trends['fr_orig'].mean("model").plot(ax=ax1,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,add_colorbar=False)
ax1.text(-0.03,0.5,"ENSEMBLE MEAN",ha='right',va='center',transform=ax1.transAxes,rotation=90)
ax2 = plt.subplot(gs[i+1,1],projection=ccrs.Miller())
fr_trends['fr_recon'].mean("model").plot(ax=ax2,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,add_colorbar=False)

ax3 = plt.subplot(gs[i+1,2])
df = fr_trends.mean("model").to_dataframe().dropna()
x = df['fr_recon'].values
y = df['fr_orig'].values
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
ax3.scatter(x,y,c=z,cmap='inferno',s=50)
ax3.set_xlim(-40,40)
ax3.set_ylim(-40,40)
ax3.plot([-40,40],[-40,40],color='black',linestyle='--')
ax3.axhline(0,color='black')
ax3.axvline(0,color='black')
r = spearmanr(df['fr_recon'],df['fr_orig'])[0]
ax3.text(0.05,0.9,f"r={np.round(r,2)}",ha='left',va='top',transform=ax3.transAxes)
ax3.set_xlabel("RECONSTRUCTED\n(%/DECADE)")
ax3.yaxis.tick_right()
ax3.set_ylabel("MODEL OUTPUT\n(%/DECADE)",rotation=-90,labelpad=30)
ax3.yaxis.set_label_position("right")
ax3.set_aspect("equal")
    
axes.append(ax1)
axes.append(ax2)
for ax in axes:
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())

sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
cax = plt.subplot(gs[-1,:2])
cbar = fig.colorbar(sm,cax=cax,orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.02)
cbar.ax.set_xlabel("FORCED MARCH SWE TREND (%/DECADE)",labelpad=5)
plt.savefig(os.path.join(project_dir,'figures','ed_fig9.jpg'),bbox_inches='tight',dpi=400)


# In[ ]:




