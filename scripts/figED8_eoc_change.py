#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xarray as xr
import geopandas as gpd
import pymannkendall as mk
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import seaborn as sns


# In[4]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope
    except:
        return np.nan


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')


# In[46]:


swe_obs = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs.nc'))
swe_obs = swe_obs.where(swe_obs['swe'].min(['product','time'])>0,drop=True)


# In[47]:


swe_hist_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc'))
swe_hist_trends['delta_swe_hist'] = 40*swe_hist_trends['swe_trend_pct'] # total change over 40-year period
swe_hist_trends['swe_hist_agree'] = (np.sign(swe_hist_trends['delta_swe_hist'])==-1).astype(int).mean("combo") # agreement on direction

swe_fut = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','swe_fut.nc'))
swe_fut = swe_fut.rename({"delta_swe":"delta_swe_fut"})
swe_fut['swe_fut_agree'] = (np.sign(swe_fut['delta_swe_fut'])==1).astype(int).mean("combo") # agreement on direction

swe_ds = xr.merge([swe_hist_trends[['delta_swe_hist','swe_hist_agree']],
                   swe_fut[['delta_swe_fut','swe_fut_agree']]]).sel(basin=swe_obs['basin'])


# In[48]:


tmean_obs = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','ndjfm_tmean_obs.nc'))
delta_tmean_obs = 40*xr.apply_ufunc(theil_sen,tmean_obs['tmean'],input_core_dims=[['time']],vectorize=True)
delta_tmean_obs.name = 'delta_tmean_hist'
delta_tmean_obs = delta_tmean_obs.to_dataset()
delta_tmean_obs['tmean_hist_agree'] = (np.sign(delta_tmean_obs['delta_tmean_hist'])==1).astype(int).mean("product")

tmean_cmip6 = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','ndjfm_tmean_cmip6.nc'))
tmean_cmip6 = tmean_cmip6.where((tmean_cmip6['time.month']>=11)|(tmean_cmip6['time.month']<=3)).resample(time='AS-NOV').mean()

# EOC change relative to historical period
delta_tmean_fut = tmean_cmip6['tmean'].sel(time=slice("2069","2098")).mean("time")-tmean_cmip6['tmean'].sel(time=slice("1980","2019")).mean("time")
delta_tmean_fut = delta_tmean_fut.dropna("model")
delta_tmean_fut.name = 'delta_tmean_fut'
delta_tmean_fut = delta_tmean_fut.to_dataset()
delta_tmean_fut['tmean_fut_agree'] = (np.sign(delta_tmean_fut['delta_tmean_fut'])==1).astype(int).mean("model")

tmean_ds = xr.merge([delta_tmean_obs,delta_tmean_fut]).sel(basin=swe_obs['basin'])


# In[49]:


ppt_obs = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','ndjfm_ppt_obs.nc'))
delta_ppt_obs = 40*100*xr.apply_ufunc(theil_sen,ppt_obs['ppt'],input_core_dims=[['time']],vectorize=True)/ppt_obs['ppt'].mean("time")
delta_ppt_obs.name = 'delta_ppt_hist'
delta_ppt_obs = delta_ppt_obs.to_dataset()
delta_ppt_obs['ppt_hist_agree'] = (np.sign(delta_ppt_obs['delta_ppt_hist'])==1).astype(int).mean("product")


ppt_cmip6 = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','ndjfm_ppt_cmip6.nc'))
ppt_cmip6 = ppt_cmip6.where((ppt_cmip6['time.month']>=11)|(ppt_cmip6['time.month']<=3)).resample(time='AS-NOV').sum(min_count=5)
delta_ppt_fut = 100*ppt_cmip6['ppt'].sel(time=slice("2069","2098")).mean("time")/ppt_cmip6['ppt'].sel(time=slice("1980","2019")).mean("time")-100
delta_ppt_fut = delta_ppt_fut.dropna("model")
delta_ppt_fut.name = 'delta_ppt_fut'
delta_ppt_fut = delta_ppt_fut.to_dataset()
delta_ppt_fut['ppt_fut_agree']= (np.sign(delta_ppt_fut['delta_ppt_fut'])==1).astype(int).mean("model")

ppt_ds = xr.merge([delta_ppt_obs,delta_ppt_fut]).sel(basin=swe_obs['basin'])


# In[50]:


runoff_hist_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','amjj_q_forced.nc'))
runoff_hist_trends['delta_q_hist'] = 40*runoff_hist_trends['forced_q_trend']
runoff_hist_trends['q_hist_agree'] = (np.sign(runoff_hist_trends['delta_q_hist'])==1).astype(int).mean("combo")


runoff_fut = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','amjj_q_fut.nc'))
runoff_fut['delta_q_fut'] = 100*runoff_fut['q_fut']/runoff_fut['q_pred'].mean("time")-100
runoff_fut["q_fut_agree"] = (np.sign(runoff_fut['delta_q_fut'])==1).mean("combo")

runoff_ds = xr.merge([runoff_hist_trends[['delta_q_hist','q_hist_agree']],
                      runoff_fut[['delta_q_fut','q_fut_agree']]]).sel(basin=swe_obs['basin'])


# In[51]:


grdc_basins = gpd.read_file(os.path.join(project_dir,'data','grdc_basins'))

swe_df = swe_ds.median("combo").to_dataframe()
swe_df = grdc_basins[['MRBID','geometry']].merge(swe_df,left_on='MRBID',right_index=True)

tmean_df = tmean_ds.median("model").mean("product").to_dataframe()
tmean_df = grdc_basins[['MRBID','geometry']].merge(tmean_df,left_on='MRBID',right_index=True)

ppt_df = ppt_ds.median("model").mean("product").to_dataframe()
ppt_df = grdc_basins[['MRBID','geometry']].merge(ppt_df,left_on='MRBID',right_index=True)

runoff_df = runoff_ds.median("combo").to_dataframe()
runoff_df = grdc_basins[['MRBID','geometry']].merge(runoff_df,left_on='MRBID',right_index=True)


# In[52]:


sns.set(style='ticks',font_scale=2)

tmean_levels = np.linspace(0,8,9)
tmean_cols = plt.get_cmap("YlOrRd")(np.linspace(0.05,0.95,len(tmean_levels)))
tmean_cmap = mpl.colors.ListedColormap(tmean_cols[:-1])
tmean_cmap.set_over(tmean_cols[-1])
tmean_norm = plt.Normalize(vmin=0,vmax=8)

ppt_levels = np.linspace(-20,40,7)
ppt_dec_levels = ppt_levels[:3]
ppt_inc_levels = ppt_levels[2:]
ppt_dec_cols = plt.get_cmap('BrBG')(np.linspace(0.05,0.45,len(ppt_dec_levels)))
ppt_inc_cols = plt.get_cmap("BrBG")(np.linspace(0.55,0.95,len(ppt_inc_levels)))
ppt_cols = np.concatenate([ppt_dec_cols,ppt_inc_cols])
ppt_cmap = mpl.colors.ListedColormap(ppt_cols[1:-1])
ppt_cmap.set_under(ppt_cols[0])
ppt_cmap.set_over(ppt_cols[-1])
ppt_norm = plt.Normalize(vmin=-20,vmax=40)

swe_levels = np.linspace(-80,50,14)
swe_dec_levels = swe_levels[:-5]
swe_inc_levels = swe_levels[-6:]
swe_dec_cols = plt.get_cmap('RdBu')(np.linspace(0.05,0.45,len(swe_dec_levels)))
swe_inc_cols = plt.get_cmap("RdBu")(np.linspace(0.55,0.95,len(swe_inc_levels)))
swe_cols = np.concatenate([swe_dec_cols,swe_inc_cols])
swe_cmap = mpl.colors.ListedColormap(swe_cols[1:-1])
swe_cmap.set_under(swe_cols[0])
swe_cmap.set_over(swe_cols[-1])
swe_norm = plt.Normalize(vmin=-80,vmax=50)

runoff_levels = np.linspace(-40,40,9)
runoff_cols = plt.get_cmap("BrBG")(np.linspace(0.05,0.95,len(runoff_levels)+1))
runoff_cmap = mpl.colors.ListedColormap(runoff_cols[1:-1])
runoff_cmap.set_under(runoff_cols[0])
runoff_cmap.set_over(runoff_cols[-1])
runoff_norm = plt.Normalize(vmin=-40,vmax=40)


df_dict = {"tmean":tmean_df,
           "ppt":ppt_df,
           "swe":swe_df,
           "q":runoff_df}

cmap_dict = {"tmean":tmean_cmap,
           "ppt":ppt_cmap,
           "swe":swe_cmap,
           "q":runoff_cmap}

norm_dict = {"tmean":tmean_norm,
           "ppt":ppt_norm,
           "swe":swe_norm,
           "q":runoff_norm}

level_dict = {"tmean":tmean_levels,
           "ppt":ppt_levels,
           "swe":swe_levels,
           "q":runoff_levels}

title_dict = {"tmean":"WINTER\nTEMPERATURE",
           "ppt":"WINTER\nPRECIPITATION",
           "swe":"MARCH SWE",
           "q":"SWE-DRIVEN\nSPRING RUNOFF"}

label_dict = {"tmean":"$^{\circ}C$",
              "ppt":"%",
              "swe":"%",
              "q":"%"}
fig = plt.figure(figsize=(24,20))
gs = gridspec.GridSpec(nrows=4,ncols=2,wspace=0.03,hspace=0.01,figure=fig)

labels = ['a','b','c','d','e','f','g','h',]
for i,var in enumerate(['tmean','ppt','swe','q']):
    df = df_dict[var]
    cmap = cmap_dict[var]
    norm = norm_dict[var]
        
    ax = plt.subplot(gs[i,0],projection=ccrs.Miller())
    df[(df[f'{var}_hist_agree']<=0.2)|(df[f'{var}_hist_agree']>=0.8)].plot(ax=ax,column=f'delta_{var}_hist',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
    df[(df[f'{var}_hist_agree']>0.2)&(df[f'{var}_hist_agree']<0.8)].plot(ax=ax,column=f'delta_{var}_hist',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.set_aspect(1.346)
    ax.text(-0.05,0.5,title_dict[var],ha='center',va='center',rotation=90,transform=ax.transAxes)
    ax.text(0.01,0.975,labels[2*i],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)

    ax2 = plt.subplot(gs[i,1],projection=ccrs.Miller())
    if var == 'tmean':
        df.plot(ax=ax2,column='delta_tmean_fut',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
    else:
        df[(df[f'{var}_fut_agree']<=0.2)|(df[f'{var}_fut_agree']>=0.8)].plot(ax=ax2,column=f'delta_{var}_fut',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
        df[(df[f'{var}_fut_agree']>0.2)&(df[f'{var}_fut_agree']<0.8)].plot(ax=ax2,column=f'delta_{var}_fut',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
    ax2.coastlines("10m")
    ax2.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax2.set_aspect(1.346)
    ax2.text(0.01,0.975,labels[2*i+1],ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

    if i==0:
        ax.title.set_text("HISTORICAL")
        ax2.title.set_text("END-OF-CENTURY")
    else:
        ax.title.set_text("")
        ax2.title.set_text("")
   
    sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array([])
    if var=='tmean':
        cbar = fig.colorbar(sm,ax=[ax,ax2],orientation="vertical",drawedges=False,ticks=level_dict[var],extend='max',shrink=0.8,pad=0.02)
    else:
        cbar = fig.colorbar(sm,ax=[ax,ax2],orientation="vertical",drawedges=False,ticks=level_dict[var],extend='both',shrink=0.8,pad=0.02)
    cbar.ax.set_ylabel(label_dict[var],labelpad=30,rotation=-90)
    for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
plt.savefig(os.path.join(project_dir,'figures','ed_fig8.jpg'),bbox_inches='tight',dpi=400)


# In[ ]:




