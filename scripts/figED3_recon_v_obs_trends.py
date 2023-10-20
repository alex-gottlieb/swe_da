#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xarray as xr
import pymannkendall as mk
import numpy as np
import pandas as pd
from multiprocessing import Pool
import geopandas as gpd


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG/'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


# In[3]:


obs_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs_trends.nc')).rename({"swe":"swe_trend_pct"})
hist_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc')).sel(basin=obs_trends['basin'])
grdc_basins = gpd.read_file(os.path.join(project_dir,'data','grdc_basins'))


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr


sns.set(style='ticks',font_scale=1.8)
levels = np.arange(-25,26,5)
cols = plt.get_cmap('RdBu')(np.linspace(0,0.99,len(levels)+1))
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-25,vmax=25)

axes = []
fig = plt.figure(figsize=(30,30))
gs = gridspec.GridSpec(nrows=7,ncols=3,wspace=0.03,height_ratios=[1,1,1,1,1,1,0.2],width_ratios=[1,1,0.6],figure=fig)

labels = ['a','b','c','d','e','f',
          'g','h','i','j','k','l',
          'm','n','o','p','q','r']

for i,p in enumerate(obs_trends['product'].values):
    ax1 = plt.subplot(gs[i,0],projection=ccrs.Miller())
    ax2 = plt.subplot(gs[i,1],projection=ccrs.Miller())
    p_recons = (10*hist_trends.where(hist_trends['combo'].str.startswith(p))['swe_trend_pct']).mean("combo").to_dataframe()
    p_recons = grdc_basins[['MRBID','geometry']].merge(p_recons,left_on='MRBID',right_index=True)
    p_obs = obs_trends.sel(product=p)[['swe_trend_pct']].to_dataframe()
    p_obs = grdc_basins[['MRBID','geometry']].merge(p_obs,left_on='MRBID',right_index=True)
    p_obs.plot(ax=ax1,column='swe_trend_pct',cmap=cmap,norm=norm,edgecolor='black',lw=0.5,transform=ccrs.PlateCarree(),legend=False)
    p_recons.plot(ax=ax2,column='swe_trend_pct',cmap=cmap,norm=norm,edgecolor='black',lw=0.5,transform=ccrs.PlateCarree(),legend=False)
    if i ==0:
        ax1.title.set_text("OBSERVED")
        ax2.title.set_text("RECONSTRUCTED")
    else:
        ax1.title.set_text("")
    r,_ = pearsonr(p_recons['swe_trend_pct'],p_obs['swe_trend_pct'])
    ax1.text(-0.07,0.5,f"{p}",ha='center',va='center',rotation='vertical',rotation_mode='anchor',transform=ax1.transAxes)
    ax1.text(0.01,0.975,labels[i],ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)
    ax2.text(0.01,0.975,labels[i+6],ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

    axes.append(ax1)
    axes.append(ax2)
    
    ax3 = plt.subplot(gs[i,2])
    ax3.scatter(p_recons['swe_trend_pct'],p_obs['swe_trend_pct'],color='black',s=30)
    ax3.set_xlim(-25,25)
    ax3.set_ylim(-25,25)
    ax3.axhline(0,color='black')
    ax3.axvline(0,color='black')
    ax3.plot([-25,25],[-25,25],color='black',linestyle='--')
    ax3.set_xlabel("")
    ax3.set_ylabel("OBSERVED\n(%/DECADE)",labelpad=40,rotation=-90)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_aspect("equal")
    ax3.text(0.02,0.975,labels[i+12],ha='left',va='top',transform=ax3.transAxes,weight='bold',fontsize=24,)
    ax3.text(0.55,0.1,f"r={np.round(r,2)}",ha='left',va='bottom',transform=ax3.transAxes)

    
ax1 = plt.subplot(gs[-2,0],projection=ccrs.Miller())
ax2 = plt.subplot(gs[-2,1],projection=ccrs.Miller())
recon_em = (10*hist_trends['swe_trend_pct']).mean("combo").to_dataframe()
recon_em = grdc_basins[['MRBID','geometry']].merge(recon_em,left_on='MRBID',right_index=True)

obs_em = obs_trends['swe_trend_pct'].mean("product").to_dataframe()
obs_em = grdc_basins[['MRBID','geometry']].merge(obs_em,left_on='MRBID',right_index=True)

obs_em.plot(ax=ax1,column='swe_trend_pct',cmap=cmap,norm=norm,edgecolor='black',lw=0.5,transform=ccrs.PlateCarree(),legend=False)
recon_em.plot(ax=ax2,column='swe_trend_pct',cmap=cmap,norm=norm,edgecolor='black',lw=0.5,transform=ccrs.PlateCarree(),legend=False)
ax1.title.set_text("")
r,_ = pearsonr(recon_em['swe_trend_pct'],obs_em['swe_trend_pct'])
ax1.text(-0.07,0.5,f"Ens. Mean",ha='center',va='center',rotation='vertical',rotation_mode='anchor',transform=ax1.transAxes)
ax1.text(0.01,0.975,labels[5],ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)
ax2.text(0.01,0.975,labels[11],ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

axes.append(ax1)
axes.append(ax2)

ax3 = plt.subplot(gs[-2,2])
ax3.scatter(p_recons['swe_trend_pct'],p_obs['swe_trend_pct'],color='black',s=30)
ax3.set_xlim(-25,25)
ax3.set_ylim(-25,25)
ax3.axhline(0,color='black')
ax3.axvline(0,color='black')
ax3.plot([-25,25],[-25,25],color='black',linestyle='--')
ax3.set_ylabel("OBSERVED\n(%/DECADE)",labelpad=40,rotation=-90)
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.text(0.02,0.975,labels[-1],ha='left',va='top',transform=ax3.transAxes,weight='bold',fontsize=24,)
ax3.text(0.55,0.1,f"r={np.round(r,2)}",ha='left',va='bottom',transform=ax3.transAxes)
ax3.set_xlabel("RECONSTRUCTED\n(%/DECADE)")
ax3.set_aspect("equal")
for ax in axes:
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())

cax = plt.subplot(gs[-1,:2])
sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm,cax=cax,orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.02)
cbar.ax.set_xlabel("MARCH SWE TREND (%/DECADE)",labelpad=15)
plt.savefig(os.path.join(project_dir,'figures','ed_fig3.jpg'),bbox_inches='tight',dpi=400)
