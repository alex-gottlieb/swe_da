#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os 
import xarray as xr
import numpy as np
import pymannkendall as mk
from sklearn.metrics import mean_squared_error
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd


# In[8]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


# In[9]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope
    except:
        return np.nan


# In[10]:


q_recons = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','amjj_q_hist.nc'))
q_recons['q_obs_pct'] = 100*q_recons['q_obs']/q_recons['q_obs'].mean("time")
q_recons['q_pred_pct'] = 100*q_recons['q_pred']/q_recons['q_pred'].mean("time")

q_recon_trends = xr.apply_ufunc(theil_sen,q_recons['q_pred'],input_core_dims=[['time']],vectorize=True)
q_recon_trends.name = 'q_trend'
q_recon_trends = q_recon_trends.to_dataset()
q_recon_trends['q_trend_pct'] = 100*q_recon_trends['q_trend']/q_recons['q_pred'].mean("time")

q_obs_trends = xr.apply_ufunc(theil_sen,q_recons['q_obs'].mean("combo"),input_core_dims=[['time']],vectorize=True)
q_obs_trends.name = 'q_obs_trend'
q_obs_trends = q_obs_trends.to_dataset()
q_obs_trends['q_obs_trend_pct'] = 100*q_obs_trends['q_obs_trend']/q_recons['q_obs'].mean("combo").mean("time")


# In[17]:


grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other','grdc_basins'))
obs_trend_df = q_obs_trends.to_dataframe()
recon_trend_df = q_recon_trends.median("combo").to_dataframe()

trend_df = obs_trend_df.merge(recon_trend_df,left_index=True,right_index=True)
trend_df = grdc_basins[['MRBID','geometry']].merge(trend_df,left_on='MRBID',right_index=True)


# In[13]:


def r2(obs,pred):
    nas = np.logical_or(np.isnan(obs),np.isnan(pred))
    obs = obs[~nas]
    pred = pred[~nas]    
    try:
        return 1-np.power(obs-pred,2).sum()/np.power(obs-obs.mean(),2).sum()
    except:
        return np.nan
    
q_em_r2 = xr.apply_ufunc(r2,q_recons['q_obs_pct'].mean("combo"),q_recons['q_pred_pct'].median("combo"),input_core_dims=[['time'],['time']],vectorize=True)


# In[27]:



def rmse(obs,pred):
    nas = np.logical_or(np.isnan(obs),np.isnan(pred))
    try:
        return mean_squared_error(obs[~nas],pred[~nas],squared=False)
    except:
        return np.nan
    
recon_em_rmse = xr.apply_ufunc(rmse,q_recons['q_obs'].mean("combo"),q_recons['q_pred'].median("combo"),input_core_dims=[['time'],['time']],vectorize=True)
recon_em_nrmse = 100*recon_em_rmse/(q_recons['q_obs'].mean("combo").max("time")-q_recons['q_obs'].mean("combo").min("time"))


# In[30]:


q_em_r2.name = 'r2'
r2_df = q_em_r2.to_dataframe()
recon_em_nrmse.name = 'rmse'
rmse_df = recon_em_nrmse.to_dataframe()
skill_df = r2_df.merge(rmse_df,left_index=True,right_index=True)
skill_df = grdc_basins[['MRBID','geometry']].merge(skill_df,left_on='MRBID',right_index=True)


# In[32]:


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

fig = plt.figure(figsize=(16,12))
gs = gridspec.GridSpec(nrows=2,ncols=1,figure=fig)

ax1 = plt.subplot(gs[0],projection=ccrs.Miller())
skill_df.plot(ax=ax1,column='r2',cmap=r2_cmap,norm=r2_norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)

ax1.coastlines("10m")
ax1.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax1.text(0.02,0.95,'a',ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)

ax1_ins = inset_axes(ax1,width="20%",height='20%',
               loc='lower center',
                bbox_to_anchor=(-0.1,0.07,1,1),
                bbox_transform=ax1.transAxes)
bins = np.arange(0,1.01,0.1)
r2_counts = skill_df['r2'].groupby(pd.cut(skill_df['r2'],bins,labels=bins[:-1]+0.025)).size()
r2_counts = r2_counts/r2_counts.sum()
ax1_ins.bar(r2_counts.index,r2_counts,width=0.1)
ax1_ins.axvline(skill_df['r2'].median(),color='red',)
ax1_ins.text(0.05,0.85,np.round(skill_df['r2'].median(),2),ha='left',va='top',color='red',fontsize=10,transform=ax1_ins.transAxes)
ax1_ins.set_ylim(0,0.4)
ax1_ins.set_xticks([0,0.5,1])
ax1_ins.set_xlim(0,1)
ax1_ins.set_yticks([])
ax1_ins.tick_params(labelsize=10,length=0.2)
for tl in ax1_ins.get_xticklabels():
    tl.set_backgroundcolor('white')
    
ax2 = plt.subplot(gs[1],projection=ccrs.Miller())
skill_df.plot(ax=ax2,column='rmse',cmap=rmse_cmap,norm=rmse_norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax2.coastlines("10m")
ax2.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax2.text(0.02,0.95,'b',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

ax2_ins = inset_axes(ax2,width="20%",height='20%',
               loc='lower center',
                bbox_to_anchor=(-0.1,0.07,1,1),
                bbox_transform=ax2.transAxes)
bins = np.arange(0,51,5)
rmse_counts = skill_df['rmse'].groupby(pd.cut(skill_df['rmse'],bins,labels=bins[:-1]+5)).size()
rmse_counts = rmse_counts/rmse_counts.sum()
ax2_ins.bar(rmse_counts.index,rmse_counts,width=5)
ax2_ins.axvline(skill_df['rmse'].median(),color='red',)
ax2_ins.text(0.9,0.85,np.round(skill_df['rmse'].median(),1),ha='right',va='top',color='red',fontsize=10,transform=ax2_ins.transAxes)
ax2_ins.set_ylim(0,0.7)
ax2_ins.set_xticks(np.arange(0,51,10))
ax2_ins.set_xlim(0,50)
ax2_ins.set_yticks([])
ax2_ins.tick_params(labelsize=10,length=0.2)
for tl in ax2_ins.get_xticklabels():
    tl.set_backgroundcolor('white')
# cax1 = plt.subplot(gs[-1,0])
r2_sm = plt.cm.ScalarMappable(norm=r2_norm,cmap=r2_cmap)
r2_sm.set_array([])
r2_cbar = fig.colorbar(r2_sm,ax=ax1,orientation="horizontal",drawedges=False,ticks=r2_levels[::2],extend='min',shrink=0.7,pad=0.03)
r2_cbar.ax.set_xlabel(r"$R^2$",labelpad=10)

# cax2 = plt.subplot(gs[-1,1])
rmse_sm = plt.cm.ScalarMappable(norm=rmse_norm,cmap=rmse_cmap)
rmse_sm.set_array([])
rmse_cbar = fig.colorbar(rmse_sm,ax=ax2,orientation="horizontal",drawedges=False,ticks=rmse_levels,extend='max',shrink=0.7,pad=0.03)
rmse_cbar.ax.set_xlabel(r"RMSE (%)",labelpad=10)
plt.savefig(os.path.join(project_dir,'figure','ed_fig10ab.jpg'),bbox_inches='tight',dpi=400)
plt.show()


# In[19]:


sns.set(style='ticks',font_scale=1.8)

levels = np.arange(-3,1.1,0.5)
dec_levels = levels[:-2]
inc_levels = levels[-3:]
dec_cols = plt.get_cmap('BrBG')(np.linspace(0.05,0.45,len(dec_levels)))
inc_cols = plt.get_cmap("BrBG")(np.linspace(0.55,0.95,len(inc_levels)))
cols = np.concatenate([dec_cols,inc_cols])
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-3,vmax=1)



fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=3,ncols=2,hspace=0.15,height_ratios=[1,1,0.1],width_ratios=[0.55,0.45],figure=fig)

ax1 = plt.subplot(gs[0,0],projection=ccrs.Miller())
trend_df.plot(column='q_obs_trend_pct',ax=ax1,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,edgecolor='black',lw=0.5,legend=False)
ax1.title.set_text("OBSERVED")
ax1.text(0.01,0.975,'c',ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)

ax2 = plt.subplot(gs[1,0],projection=ccrs.Miller())
trend_df.plot(column='q_trend_pct',ax=ax2,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,edgecolor='black',lw=0.5,legend=False)
ax2.title.set_text("RECONSTRUCTED")
ax2.text(0.01,0.975,'d',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

for ax in [ax1,ax2]:
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
 
ax3 = plt.subplot(gs[:2,1])
trend_df.plot.scatter(ax=ax3,x='q_trend_pct',y='q_obs_trend_pct',color='black',s=20)
ax3.axhline(0,color='black')
ax3.axvline(0,color='black')
ax3.plot([-3,1],[-3,1],color='black',linestyle='--')
ax3.set_xlim(-3,1)
ax3.set_ylim(-3,1)
ax3.set_xticks(np.linspace(-3,1,9))
ax3.set_yticks(np.linspace(-3,1,9))
ax3.set_xticklabels(np.linspace(-30,10,9).astype(int))
ax3.set_yticklabels(np.linspace(-30,10,9).astype(int))

ax3.set_ylabel("OBSERVED (%/DECADE)")
ax3.set_xlabel("RECONSTRUCTED (%/DECADE)")
ax3.set_aspect("equal")
ax3.text(0.15,0.6,'r=0.94',ha='left',va='center',transform=ax3.transAxes)
ax3.text(0.025,0.975,'e',ha='left',va='top',transform=ax3.transAxes,weight='bold',fontsize=24,)

sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
cax = plt.subplot(gs[2,0])
cbar = fig.colorbar(sm,cax=cax,orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.02)
cbar.ax.set_xlabel("SPRING RUNOFF TREND (%/DECADE)",labelpad=15)
cbar.ax.set_xticklabels((10*levels).astype(int))
plt.savefig(os.path.join(project_dir,'figure','ed_fig10cde.jpg'),bbox_inches='tight',dpi=400)
plt.show()


# In[ ]:




