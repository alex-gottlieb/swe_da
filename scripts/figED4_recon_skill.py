#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd


# In[ ]:





# In[6]:


pd.read_csv(recon_files[0])


# In[15]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG/'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')

obs = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs.nc'))
obs = obs.where(obs['swe'].min(['product','time'])>0,drop=True)

recon_dir = os.path.join(project_dir,'data','basin_scale','hist_recons')
recon_files = [os.path.join(recon_dir,f) for f in os.listdir(recon_dir)]
recon_files.sort()

recons = xr.concat([xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['basin','time'])).assign_coords(combo=f.split("/")[-1].split(".")[0]) for f in recon_files], dim='combo')
recons = recons.sel(basin=obs['basin'])
recons['swe_obs_pct'] = recons['swe_obs']/recons['swe_obs'].mean('time')
recons['swe_pred_pct'] = recons['swe_pred']/recons['swe_pred'].mean('time')

grdc_basins = gpd.read_file(os.path.join(project_dir,'data','grdc_basins'))


# In[24]:


def r2(obs,pred):
    try:
        return 1-np.power(obs-pred,2).sum()/np.power(obs-obs.mean(),2).sum()
    except:
        return np.nan

def rmse(obs,pred):
    nas = np.logical_or(np.isnan(obs),np.isnan(pred))
    try:
        return mean_squared_error(obs[~nas],pred[~nas],squared=False)
    except:
        return np.nan
    
# R-squared and NRMSE for each product
all_p_r2 = []
all_p_rmse = []

for p in np.unique(obs['product']):
    p_ds = recons.where(recons['combo'].str.startswith(p),drop=True)
    p_em_r2 = xr.apply_ufunc(r2,p_ds['swe_obs_pct'].mean("combo"),p_ds['swe_pred_pct'].median("combo"),input_core_dims=[['time'],['time']],vectorize=True)  
    p_em_r2 = p_em_r2.assign_coords(product=p)
    
    p_em_rmse = xr.apply_ufunc(rmse,p_ds['swe_obs'].mean("combo"),p_ds['swe_pred'].median("combo"),input_core_dims=[['time'],['time']],vectorize=True) 
    p_em_rmse = 100*p_em_rmse/(p_ds['swe_obs'].mean("combo").max("time")-p_ds['swe_obs'].mean("combo").min("time")) # normalize by observed range
    p_em_rmse = p_em_rmse.assign_coords(product=p)
    all_p_r2.append(p_em_r2)
    all_p_rmse.append(p_em_rmse)

all_p_r2 = xr.concat(all_p_r2,dim='product')
all_p_r2.name = 'r2'

all_p_rmse = xr.concat(all_p_rmse,dim='product')
all_p_rmse.name = 'rmse'


# In[25]:


r2_df = all_p_r2.to_dataframe()
rmse_df = all_p_rmse.to_dataframe()
skill_df = r2_df.merge(rmse_df,left_index=True,right_index=True)


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style='ticks',font_scale=1.7)
r2_levels = np.arange(0.5,1.01,0.05)
r2_cols = plt.get_cmap('YlGnBu')(np.linspace(0.1,0.9,len(r2_levels)))
r2_cmap = mpl.colors.ListedColormap(r2_cols[1:])
r2_cmap.set_under(r2_cols[0])
r2_norm = plt.Normalize(vmin=0.5,vmax=1)

rmse_levels = np.arange(0,51,5)
rmse_cols = plt.get_cmap('YlOrRd')(np.linspace(0,0.99,len(rmse_levels)))
rmse_cmap = mpl.colors.ListedColormap(rmse_cols[:-1])
rmse_cmap.set_over(rmse_cols[-1])
rmse_norm = plt.Normalize(vmin=0,vmax=50)

fig = plt.figure(figsize=(16,20))
gs = gridspec.GridSpec(nrows=5,ncols=2,wspace=0.05,figure=fig)
r2_axes = []
rmse_axes = []
r2_labels = ['a','b','c','d','e']
rmse_labels = ['f','g','h','i','j']
# rmse_labels = ['i','j','k','l','m','n','o','p']
for i, p in enumerate(np.unique(obs['product'])):
    ax1 = plt.subplot(gs[i,0],projection=ccrs.Miller())
    p_df = skill_df.loc[p]
    p_df = grdc_basins[['MRBID','geometry']].merge(p_df,left_on='MRBID',right_on='basin') 
    p_df.plot(ax=ax1,column='r2',cmap=r2_cmap,norm=r2_norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)

    ax1.text(-0.04,0.5,p,ha='center',va='center',rotation='vertical',rotation_mode='anchor',transform=ax1.transAxes)
    ax1.coastlines("10m")
    ax1.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax1.text(0.02,0.95,r2_labels[i],ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=20,)

    ax1_ins = inset_axes(ax1,width="20%",height='20%',
                   loc='lower center',
                    bbox_to_anchor=(-0.1,0.07,1,1),
                    bbox_transform=ax1.transAxes)
    bins = np.arange(0,1.01,0.1)
    r2_counts = p_df['r2'].groupby(pd.cut(p_df['r2'],bins,labels=bins[:-1]+0.025)).size()
    r2_counts = r2_counts/r2_counts.sum()
    ax1_ins.bar(r2_counts.index,r2_counts,width=0.1)
    ax1_ins.axvline(p_df['r2'].median(),color='red',)
    ax1_ins.text(0.05,0.85,np.round(p_df['r2'].median(),2),ha='left',va='top',color='red',fontsize=10,transform=ax1_ins.transAxes)
    ax1_ins.set_ylim(0,0.4)
    ax1_ins.set_xticks([0,0.5,1])
    ax1_ins.set_xlim(0,1)
    ax1_ins.set_yticks([])
    ax1_ins.tick_params(labelsize=10,length=0.2)
    for tl in ax1_ins.get_xticklabels():
        tl.set_backgroundcolor('white')
    r2_axes.append(ax1)

    ax2 = plt.subplot(gs[i,1],projection=ccrs.Miller())
    p_df.plot(ax=ax2,column='rmse',cmap=rmse_cmap,norm=rmse_norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
    ax2.coastlines("10m")
    ax2.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax2.text(0.02,0.95,rmse_labels[i],ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=20,)
    
    ax2_ins = inset_axes(ax2,width="20%",height='20%',
                   loc='lower center',
                    bbox_to_anchor=(-0.1,0.07,1,1),
                    bbox_transform=ax2.transAxes)
    bins = np.arange(0,51,5)
    rmse_counts = p_df['rmse'].groupby(pd.cut(p_df['rmse'],bins,labels=bins[:-1]+5)).size()
    rmse_counts = rmse_counts/rmse_counts.sum()
    ax2_ins.bar(rmse_counts.index,rmse_counts,width=5)
    ax2_ins.axvline(p_df['rmse'].median(),color='red',)
    ax2_ins.text(0.9,0.85,np.round(p_df['rmse'].median(),1),ha='right',va='top',color='red',fontsize=10,transform=ax2_ins.transAxes)
    ax2_ins.set_ylim(0,0.7)
    ax2_ins.set_xticks(np.arange(0,51,10))
    ax2_ins.set_xlim(0,50)
    ax2_ins.set_yticks([])
    ax2_ins.tick_params(labelsize=10,length=0.2)
    for tl in ax2_ins.get_xticklabels():
        tl.set_backgroundcolor('white')
    rmse_axes.append(ax2)
    
# cax1 = plt.subplot(gs[-1,0])
r2_sm = plt.cm.ScalarMappable(norm=r2_norm,cmap=r2_cmap)
r2_sm.set_array([])
r2_cbar = fig.colorbar(r2_sm,ax=r2_axes,orientation="horizontal",drawedges=False,ticks=r2_levels[::2],extend='min',shrink=0.9,pad=0.01)
r2_cbar.ax.set_xlabel(r"$R^2$",labelpad=15)

# cax2 = plt.subplot(gs[-1,1])
rmse_sm = plt.cm.ScalarMappable(norm=rmse_norm,cmap=rmse_cmap)
rmse_sm.set_array([])
rmse_cbar = fig.colorbar(rmse_sm,ax=rmse_axes,orientation="horizontal",drawedges=False,ticks=rmse_levels,extend='max',shrink=0.9,pad=0.01)
rmse_cbar.ax.set_xlabel(r"RMSE (%)",labelpad=15)
plt.savefig(os.path.join(project_dir,'figures','ed_fig4.jpg'),bbox_inches='tight',dpi=400)
plt.show()


# In[ ]:




