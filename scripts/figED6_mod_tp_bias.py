#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import xarray as xr
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import seaborn as sns


# In[ ]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


# In[ ]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope
    except:
        return np.nan


# In[ ]:


# obs tmean trends
tmean_dir = os.path.join(project_dir,'data','regrid_2deg','tmean')
tmean_files = [os.path.join(tmean_dir,f) for f in os.listdir(tmean_dir)]
tmean_files.sort()

tmean_trends = []
for f in tmean_files:
    mod = f.split("/")[-1].split(".")[0]
    with xr.open_dataset(f) as ds:
        ds_ndjfm = ds.where((ds['time.month']>=11)|(ds['time.month']<=3)).resample(time='AS-NOV').mean()
        ds_ndjfm = ds_ndjfm.sel(time=slice("1980","2019"))
        trend = xr.apply_ufunc(theil_sen,ds_ndjfm['tmean'],input_core_dims=[['time']],vectorize=True)
        trend.name = 'tmean_trend'
        trend_ds = trend.to_dataset()
        trend_ds = trend_ds.assign_coords(product=mod)
        tmean_trends.append(trend_ds)
        print(mod)
tmean_trends = xr.concat(tmean_trends,dim='product')


# In[ ]:


# obs precip trends
ppt_dir = os.path.join(project_dir,'data','regrid_2deg','ppt')
ppt_files = [os.path.join(ppt_dir,f) for f in os.listdir(ppt_dir)]
ppt_files.sort()

ppt_trends = []
for f in ppt_files:
    mod = f.split("/")[-1].split(".")[0]
    with xr.open_dataset(f) as ds:
        if mod =='MSWEP':
            ds = ds.rename({"p":"ppt"})
        ds_ndjfm = ds.where((ds['time.month']>=11)|(ds['time.month']<=3)).resample(time='AS-NOV').sum()
        ds_ndjfm = ds_ndjfm.sel(time=slice("1980","2019"))
        trend = xr.apply_ufunc(theil_sen,ds_ndjfm['ppt'],input_core_dims=[['time']],vectorize=True)
        trend.name = 'ppt_trend'
        trend_ds = trend.to_dataset()
        trend_ds['ppt_trend_pct'] = 100*trend_ds['ppt_trend']/ds_ndjfm['ppt'].mean("time")
        trend_ds = trend_ds.assign_coords(product=mod)
        ppt_trends.append(trend_ds)
ppt_trends = xr.concat(ppt_trends,dim='product')


# In[ ]:


mods = ['ACCESS-CM2',
 'ACCESS-ESM1-5',
 'BCC-CSM2-MR',
 'CNRM-CM6-1',
 'CanESM5',
 'GFDL-CM4',
 'GFDL-ESM4',
 'IPSL-CM6A-LR',
 'MIROC6',
 'MRI-ESM2-0',
 'NorESM2-LM']


# In[ ]:


# cmip6 ppt trends
mod_ppt_dir = os.path.join(project_dir,'trad_da','data','cmip6','ppt_hist')
mod_ppt_files = [os.path.join(mod_ppt_dir,f) for f in os.listdir(mod_ppt_dir)]
mod_ppt_files = [f for f in mod_ppt_files if f.split("/")[-1].split("_")[0] in mods]
mod_ppt_files.sort()

mod_ppt_trends = []
for f in mod_ppt_files:
    mod = f.split("/")[-1].split(".")[0]
    with xr.open_dataset(f) as ds:
        ds_ndjfm = ds.where((ds['time.month']>=11)|(ds['time.month']<=3)).resample(time='AS-NOV').sum()
        ds_ndjfm = ds_ndjfm.sel(time=slice("1980","2019"))
        trend = xr.apply_ufunc(theil_sen,ds_ndjfm['ppt'],input_core_dims=[['time']],vectorize=True)
        trend.name = 'ppt_trend'
        trend_ds = trend.to_dataset()
        trend_ds['ppt_trend_pct'] = 100*trend_ds['ppt_trend']/ds_ndjfm['ppt'].mean("time")
        trend_ds = trend_ds.assign_coords(model=mod)
        mod_ppt_trends.append(trend_ds)
        print(f)
mod_ppt_trends = xr.concat(mod_ppt_trends,dim='model')


# In[ ]:


# cmip6 tmean trends
mod_tmean_dir = os.path.join(project_dir,'trad_da','data','cmip6','tmean_hist')
mod_tmean_files = [os.path.join(mod_tmean_dir,f) for f in os.listdir(mod_tmean_dir)]
mod_tmean_files = [f for f in mod_tmean_files if f.split("/")[-1].split("_")[0] in mods]
mod_tmean_files.sort()

mod_tmean_trends = []
for f in mod_tmean_files:
    mod = f.split("/")[-1].split(".")[0]
    with xr.open_dataset(f) as ds:
        ds_ndjfm = ds.where((ds['time.month']>=11)|(ds['time.month']<=3)).resample(time='AS-NOV').mean()
        ds_ndjfm = ds_ndjfm.sel(time=slice("1980","2019"))
        trend = xr.apply_ufunc(theil_sen,ds_ndjfm['tmean'],input_core_dims=[['time']],vectorize=True)
        trend.name = 'tmean_trend'
        trend_ds = trend.to_dataset()
        trend_ds = trend_ds.assign_coords(model=mod)
        mod_tmean_trends.append(trend_ds)
        print(f)
mod_tmean_trends = xr.concat(mod_tmean_trends,dim='model')


# In[4]:


gl_mask = xr.open_dataset(os.path.join(project_dir,'data','regrid_2deg','gl_mask.nc'))
swe_mask = xr.open_dataset(os.path.join(project_dir,'data','regrid_2deg','swe_mask.nc'))

obs_tmean_trends = obs_tmean_trends.where((gl_mask['gl_mask']==1)&(swe_mask['swe']==1))
obs_ppt_trends = obs_ppt_trends.where((gl_mask['gl_mask']==1)&(swe_mask['swe']==1))

mod_tmean_trends = mod_tmean_trends.where((gl_mask['gl_mask']==1)&(swe_mask['swe']==1))
mod_ppt_trends = mod_ppt_trends.where((gl_mask['gl_mask']==1)&(swe_mask['swe']==1))


# In[9]:


sns.set(style='ticks',font_scale=2)

tmean_levels = np.linspace(-0.1,0.1,11)
tmean_cols = plt.get_cmap("RdBu_r")(np.linspace(0.05,0.95,len(tmean_levels)+1))
tmean_cmap = mpl.colors.ListedColormap(tmean_cols[1:-1])
tmean_cmap.set_under(tmean_cols[0])
tmean_cmap.set_over(tmean_cols[-1])
tmean_norm = plt.Normalize(vmin=-0.1,vmax=0.1)

fig = plt.figure(figsize=(28,20))
gs = gridspec.GridSpec(nrows=3,ncols=2,hspace=0.01,wspace=0.05,figure=fig)
ax1 = plt.subplot(gs[0,0],projection=ccrs.Miller())
obs_tmean_trends['tmean_trend'].mean("product").plot(ax=ax1,transform=ccrs.PlateCarree(),cmap=tmean_cmap,norm=tmean_norm,add_colorbar=False)
ax1.coastlines()
ax1.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax1.title.set_text("WINTER TEMPERATURE")
ax1.text(-0.03,0.5,"OBSERVATIONS",ha='right',va='center',rotation=90,transform=ax1.transAxes)
ax1.text(0.01,0.975,'a',ha='left',va='top',transform=ax1.transAxes,weight='bold',fontsize=24,)

ax2 = plt.subplot(gs[1,0],projection=ccrs.Miller())
mod_tmean_trends['tmean_trend'].mean("model").plot(ax=ax2,transform=ccrs.PlateCarree(),cmap=tmean_cmap,norm=tmean_norm,add_colorbar=False)
ax2.coastlines()
ax2.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax2.title.set_text("")
ax2.text(-0.05,0.5,"CMIP6 HISTORICAL\nENSEMBLE MEAN",ha='center',va='center',rotation=90,transform=ax2.transAxes)
ax2.text(0.01,0.975,'b',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

ax3 = plt.subplot(gs[2,0],projection=ccrs.Miller())
(mod_tmean_trends['tmean_trend']-obs_tmean_trends['tmean_trend']).mean(['product','model']).plot(ax=ax3,transform=ccrs.PlateCarree(),cmap=tmean_cmap,norm=tmean_norm,add_colorbar=False)
tmean_hatch = (obs_tmean_trends['tmean_trend'].mean("product")>mod_tmean_trends['tmean_trend']).mean("model")
tmean_hatch = tmean_hatch.where((mask['gl_mask']>0.9)&(swe_mask==1))
tmean_hatch.plot.contourf(ax=ax3,transform=ccrs.PlateCarree(),colors='none',levels=[-np.inf,0.025,0.975,np.inf],hatches=['..',None,'..'],add_colorbar=False)
ax3.coastlines()
ax3.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax3.title.set_text("")
ax3.text(-0.05,0.5,"DIFFERENCE\n(CMIP - OBS)",ha='center',va='center',rotation=90,transform=ax3.transAxes)
ax3.text(0.01,0.975,'c',ha='left',va='top',transform=ax3.transAxes,weight='bold',fontsize=24,)

tmean_sm = plt.cm.ScalarMappable(norm=tmean_norm,cmap=tmean_cmap)
tmean_sm.set_array([])
tmean_cbar = fig.colorbar(tmean_sm,ax=[ax1,ax2,ax3],orientation="horizontal",drawedges=False,ticks=tmean_levels,extend='both',shrink=0.9,pad=0.02)
tmean_cbar.ax.set_xlabel("$^{\circ}C$/DECADE",labelpad=15)
tmean_cbar.ax.set_xticklabels(np.round(10*tmean_levels,1))

ppt_levels = np.linspace(-1,1,11)
ppt_cols = plt.get_cmap("BrBG")(np.linspace(0.05,0.95,len(ppt_levels)+1))
ppt_cmap = mpl.colors.ListedColormap(ppt_cols[1:-1])
ppt_cmap.set_under(ppt_cols[0])
ppt_cmap.set_over(ppt_cols[-1])
ppt_norm = plt.Normalize(vmin=-1,vmax=1)

ax4 = plt.subplot(gs[0,1],projection=ccrs.Miller())
obs_ppt_trends['ppt_trend_pct'].mean("product").plot(ax=ax4,transform=ccrs.PlateCarree(),cmap=ppt_cmap,norm=ppt_norm,add_colorbar=False)
ax4.coastlines()
ax4.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax4.title.set_text("WINTER PRECIPITATION")
ax4.text(0.01,0.975,'d',ha='left',va='top',transform=ax4.transAxes,weight='bold',fontsize=24,)

ax5 = plt.subplot(gs[1,1],projection=ccrs.Miller())
mod_ppt_trends['ppt_trend_pct'].mean("model").plot(ax=ax5,transform=ccrs.PlateCarree(),cmap=ppt_cmap,norm=ppt_norm,add_colorbar=False)
ax5.coastlines()
ax5.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax5.title.set_text("")
ax5.text(0.01,0.975,'e',ha='left',va='top',transform=ax5.transAxes,weight='bold',fontsize=24,)

ax6 = plt.subplot(gs[2,1],projection=ccrs.Miller())
(mod_ppt_trends['ppt_trend_pct']-obs_ppt_trends['ppt_trend_pct']).mean(['product','model']).plot(ax=ax6,transform=ccrs.PlateCarree(),cmap=ppt_cmap,norm=ppt_norm,add_colorbar=False)
ppt_hatch = (obs_ppt_trends['ppt_trend_pct'].mean("product")>mod_ppt_trends['ppt_trend_pct']).mean("model")
ppt_hatch = ppt_hatch.where((mask['gl_mask']>0.9)&(swe_mask==1))
ppt_hatch.plot.contourf(ax=ax6,transform=ccrs.PlateCarree(),colors='none',levels=[-np.inf,0.025,0.975,np.inf],hatches=['////',None,'////'],add_colorbar=False)
ax6.coastlines()
ax6.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax6.title.set_text("")
ax6.text(0.01,0.975,'f',ha='left',va='top',transform=ax6.transAxes,weight='bold',fontsize=24,)

ppt_sm = plt.cm.ScalarMappable(norm=ppt_norm,cmap=ppt_cmap)
ppt_sm.set_array([])
ppt_cbar = fig.colorbar(ppt_sm,ax=[ax4,ax5,ax6],orientation="horizontal",drawedges=False,ticks=ppt_levels,extend='both',shrink=0.9,pad=0.02)
ppt_cbar.ax.set_xlabel("%/DECADE",labelpad=15)
plt.savefig(os.path.join(project_dir,'figures','ed_fig6.jpg'),bbox_inches='tight',dpi=400)

