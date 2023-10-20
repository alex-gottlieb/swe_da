#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import xarray as xr
import pymannkendall as mk
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd
from shapely.geometry import Point
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")


# In[ ]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG/'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


obs = pd.read_csv(os.path.join(project_dir,'data','in_situ','all_in_situ.csv'))
obs['station_id'] = obs['station_id'].astype(str)
obs['time'] = obs['time'].astype('datetime64[ns]')
obs_meta = obs.groupby("station_id").first()[['lat','lon']]
obs_meta['geometry'] = obs_meta.apply(lambda row: Point(row['lon'],row['lat']),axis=1)
obs_meta = gpd.GeoDataFrame(obs_meta)

keep_stns = obs.groupby("station_id").count()['swe']>=35
keep_stns = keep_stns[keep_stns].index

obs_ds = xr.Dataset.from_dataframe(obs.set_index(['station_id','time'])[['swe']])
recon_dir = os.path.join(project_dir,'data','in_situ','hist_recons')
recon_files = [os.path.join(recon_dir,f) for f in os.listdir(recon_dir) if f.endswith("csv")]
recon_files.sort()

recons = [xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['station_id','time']))[['swe_pred']].assign_coords(combo=f.split("/")[-1].split(".")[0]) for f in recon_files]
recons = xr.concat(recons,dim='combo')
recons['time'] = recons['time'].astype('datetime64[ns]')
recons = recons.sel(time=slice("1980","2019"),station_id=keep_stns)


# In[ ]:


combined = xr.merge([obs_ds,recons],join='inner')


# In[33]:


def r2(obs,pred):
    nas = np.logical_or(np.isnan(obs),np.isnan(pred))
    obs = obs[~nas]
    pred = pred[~nas]    
    try:
        return 1-np.power(obs-pred,2).sum()/np.power(obs-obs.mean(),2).sum()
    except:
        return np.nan
    
def rmse(obs,pred):
    nas = np.logical_or(np.isnan(obs),np.isnan(pred))
    obs = obs[~nas]
    pred = pred[~nas]
    try:
        return 100*mean_squared_error(obs,pred,squared=False)/(obs.max()-obs.min())
    except:
        return np.nan
    
# calcluate R-squared and normalized RMSE for each site
in_situ_r2 = xr.apply_ufunc(r2,combined['swe'],combined['swe_pred'],input_core_dims=[['time'],['time']],vectorize=True)
in_situ_em_r2 = xr.apply_ufunc(r2,combined['swe'],combined['swe_pred'].where(in_situ_r2>0).mean("combo"),input_core_dims=[['time'],['time']],vectorize=True)

in_situ_rmse = xr.apply_ufunc(rmse,combined['swe'],combined['swe_pred'],input_core_dims=[['time'],['time']],vectorize=True)
in_situ_em_rmse = xr.apply_ufunc(rmse,combined['swe'],combined['swe_pred'].where(in_situ_r2>0).mean("combo"),input_core_dims=[['time'],['time']],vectorize=True)
in_situ_em_rmse = 100*in_situ_em_rmse/(combined['swe'].max('time')-combined['swe'].min("time"))

in_situ_em_r2.name = 'r2'
in_situ_em_rmse.name = 'rmse'
skill_ds = xr.merge([in_situ_em_r2,in_situ_em_rmse]).sel(station_id=keep_stns)
skill_df = skill_ds.to_dataframe()
skill_df = obs_meta.merge(skill_df,left_index=True,right_index=True)


# In[48]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope, out.intercept
    except:
        return np.nan, np.nan
    
    
obs_trend,_ = xr.apply_ufunc(theil_sen,combined['swe'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
obs_trend = 100*obs_trend/combined['swe'].mean("time")
recon_trend,_ = xr.apply_ufunc(theil_sen,combined['swe_pred'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
recon_trend = 100*recon_trend/combined['swe_pred'].mean("time")


# In[ ]:


# NOTE: difficult to get all 5 subplots to arrange nicely in python, so save ab and cde separate to combine in Inkscape


# In[ ]:


sns.set(style='ticks',font_scale=1.8)
r2_levels = np.arange(0,1.01,0.1)
r2_cols = plt.get_cmap('YlGnBu')(np.linspace(0,1,len(r2_levels)))
r2_cmap = mpl.colors.ListedColormap(r2_cols[1:])
r2_cmap.set_under("grey")
r2_norm = plt.Normalize(vmin=0,vmax=1)

rmse_levels = np.arange(0,71,10)
rmse_cols = plt.get_cmap('YlOrRd')(np.linspace(0,1,len(rmse_levels)))
rmse_cmap = mpl.colors.ListedColormap(rmse_cols[:-1])
rmse_cmap.set_over(rmse_cols[-1])
rmse_norm = plt.Normalize(vmin=0,vmax=70)

fig = plt.figure(figsize=(16,12))
gs = gridspec.GridSpec(nrows=2,ncols=1,figure=fig)
ax1 = plt.subplot(gs[0],projection=ccrs.Miller())
skill_df.plot(ax=ax1,column='r2',transform=ccrs.PlateCarree(),cmap=r2_cmap,norm=r2_norm,legend=False)
ax1.text(0.015,0.975,'a',ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)
ax1_ins = inset_axes(ax1,width="25%",height='20%',
               loc='lower center',
                bbox_to_anchor=(-0.1,0.07,1,1),
                bbox_transform=ax1.transAxes)
bins = np.arange(0,1.01,0.1)
r2_counts = skill_df['r2'].groupby(pd.cut(skill_df['r2'],bins,labels=bins[:-1]+0.05)).size()
r2_counts = r2_counts/r2_counts.sum()
ax1_ins.bar(r2_counts.index,r2_counts,width=0.1)
ax1_ins.axvline(skill_df['r2'].median(),color='red',)
ax1_ins.text(0.05,0.85,np.round(skill_df['r2'].median(),2),ha='left',va='top',color='red',fontsize=10,transform=ax1_ins.transAxes)
ax1_ins.set_ylim(0,0.4)
ax1_ins.set_xticks(np.arange(0,1.1,0.2))
ax1_ins.set_xlim(0,1)
ax1_ins.set_yticks([])
ax1_ins.tick_params(labelsize=10,length=0.2)
for tl in ax1_ins.get_xticklabels():
    tl.set_backgroundcolor('white')

ax2 = plt.subplot(gs[1],projection=ccrs.Miller())
skill_df.plot(ax=ax2,column='rmse',transform=ccrs.PlateCarree(),cmap=rmse_cmap,norm=rmse_norm,legend=False)
ax2.text(0.015,0.975,'b',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)
ax2_ins = inset_axes(ax2,width="25%",height='20%',
               loc='lower center',
                bbox_to_anchor=(-0.1,0.07,1,1),
                bbox_transform=ax2.transAxes)
bins = np.arange(0,71,5)
bins = np.append(bins,np.inf)
rmse_counts = skill_df['rmse'].groupby(pd.cut(skill_df['rmse'],bins,labels=bins[:-1]+2.5)).size()
rmse_counts = rmse_counts/rmse_counts.sum()
ax2_ins.bar(rmse_counts.index,rmse_counts,width=5)
ax2_ins.axvline(skill_df['rmse'].median(),color='red',)
ax2_ins.text(0.05,0.85,np.round(skill_df['rmse'].median(),1),ha='left',va='top',color='red',fontsize=10,transform=ax2_ins.transAxes)
ax2_ins.set_ylim(0,0.4)
ax2_ins.set_xticks(np.arange(0,71,10))
ax2_ins.set_xlim(0,75)
ax2_ins.set_xticklabels([0,10,20,30,40,50,60,'70+'])
ax2_ins.set_yticks([])
ax2_ins.tick_params(labelsize=10,length=0.2)
for tl in ax2_ins.get_xticklabels():
    tl.set_backgroundcolor('white')


for ax in [ax1,ax2]:
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.coastlines("10m")
    ax.add_feature(cartopy.feature.BORDERS)

r2_sm = plt.cm.ScalarMappable(norm=r2_norm,cmap=r2_cmap)
r2_sm.set_array([])
r2_cbar = fig.colorbar(r2_sm,ax=ax1,orientation="horizontal",drawedges=False,ticks=r2_levels,shrink=0.9,pad=0.02)
r2_cbar.ax.set_xlabel(r"$R^2$",labelpad=5)

rmse_sm = plt.cm.ScalarMappable(norm=rmse_norm,cmap=rmse_cmap)
rmse_sm.set_array([])
rmse_cbar = fig.colorbar(rmse_sm,ax=ax2,orientation="horizontal",drawedges=False,ticks=rmse_levels,extend='max',shrink=0.9,pad=0.02)
rmse_cbar.ax.set_xlabel(r"RMSE (%)",labelpad=5)

plt.savefig(os.path.join(project_dir,'figures','ed_fig5ab.png'),bbox_inches='tight',dpi=500)
plt.show()


# In[57]:


trend_df = xr.merge([obs_trend,recon_trend.where(in_situ_r2>0).mean("combo")]).to_dataframe()
trend_df = obs_meta.merge(trend_df,left_index=True,right_index=True)


# In[60]:


sns.set(style='ticks',font_scale=1.8)
trend_levels = np.linspace(-2,2,9)
trend_cols = plt.get_cmap('RdBu')(np.linspace(0,1,len(trend_levels)+1))
trend_cmap = mpl.colors.ListedColormap(trend_cols[1:-1])
trend_cmap.set_under(trend_cols[0])
trend_cmap.set_over(trend_cols[-1])
trend_norm = plt.Normalize(vmin=-2,vmax=2)

fig = plt.figure(figsize=(16,7))
gs = gridspec.GridSpec(nrows=3,ncols=2,hspace=0.01,height_ratios=[1,1,0.1],figure=fig)
ax = plt.subplot(gs[0,0],projection=ccrs.Miller())
trend_df.plot(ax=ax,transform=ccrs.PlateCarree(),column='swe',cmap=trend_cmap,norm=trend_norm,s=10)
ax.title.set_text("OBSERVED")
ax.coastlines("10m")
ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax.text(0.01,0.975,'c',ha='left',va='top',fontweight='bold',transform=ax.transAxes)
ax2 = plt.subplot(gs[1,0],projection=ccrs.Miller())
trend_df.plot(ax=ax2,transform=ccrs.PlateCarree(),column='swe_pred',cmap=trend_cmap,norm=trend_norm,s=10)
ax2.title.set_text("RECONSTRUCTED")
ax2.coastlines("10m")
ax2.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax2.text(0.01,0.975,'d',ha='left',va='top',fontweight='bold',transform=ax2.transAxes)

trend_sm = plt.cm.ScalarMappable(norm=trend_norm,cmap=trend_cmap)
trend_sm.set_array([])
cax = plt.subplot(gs[2,0])
trend_cbar = fig.colorbar(trend_sm,cax=cax,orientation="horizontal",drawedges=False,ticks=trend_levels,extend='both')
trend_cbar.ax.set_xlabel(r"MARCH SWE TREND (%/DECADE)",labelpad=5)
trend_cbar.ax.set_xticklabels((10*trend_levels).astype(int))

ax3 = plt.subplot(gs[:2,1])
trend_df = trend_df.dropna()
x = trend_df['swe_pred'].values
y = trend_df['swe'].values

# get point density 
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy) 
ax3.scatter(x,y,c=z,cmap='Reds',s=50)
ax3.set_xlim(-4,4)
ax3.set_ylim(-4,4)
ax3.plot([-4,4],[-4,4],color='black',linestyle='--')
ax3.set_xticks(np.linspace(-4,4,9))
ax3.set_yticks(np.linspace(-4,4,9))
ax3.set_xticklabels(np.linspace(-40,40,9).astype(int))
ax3.set_yticklabels(np.linspace(-40,40,9).astype(int))

ax3.set_aspect("equal")
ax3.axhline(0,color='black')
ax3.axvline(0,color='black')
ax3.set_xlabel("RECONSTRUCTED TREND (%/DECADE)")
ax3.set_ylabel("OBSERVED TREND (%/DECADE)")
ax3.text(0.95,0.05,'r=0.72',ha='right',va='bottom',transform=ax3.transAxes)
ax3.text(0.02,0.98,'e',ha='left',va='top',fontweight='bold',transform=ax3.transAxes)

plt.savefig(os.path.join(project_dir,'figures','ed_fig5cde.png'),bbox_inches='tight',dpi=400)


# In[ ]:




