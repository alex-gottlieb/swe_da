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
hist_dir = os.path.join(project_dir,'data','basin_scale','hist_recons')
hist_files = [os.path.join(hist_dir,f) for f in os.listdir(hist_dir)]
hist_files.sort()

noacc_dir = os.path.join(project_dir,'data','basin_scale','noacc_recons')
noacc_files = [os.path.join(noacc_dir,f) for f in os.listdir(noacc_dir)]
noacc_files.sort()

# CMIP6 models
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

noacc_files = [f for f in noacc_files if f.split("_")[-2] in mods]


# In[3]:


def theil_sen(ts):
    try:
        out=mk.original_test(ts)
        return out.slope
    except:
        return np.nan

# function for calculating trends in parallel to speed things up
def calc_trend_mp(f,hem=False):
    with xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['basin','time'])) as ds:
        ds = ds.assign_coords(combo=f.split("/")[-1].split(".")[0])
        if hem:
            ds = ds.sum("basin")
        all_trends = []
        for var in list(ds.data_vars):
            trends = xr.apply_ufunc(theil_sen,ds[var],input_core_dims=[['time']],output_core_dims=[[]],vectorize=True)
            trends = trends/ds[var].mean("time")
            trends = 100*trends
            trends.name = f'{var}_trend_pct'
            all_trends.append(trends)
        all_trends = xr.merge(all_trends)
    return all_trends


# In[28]:


if not os.path.exists(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc')):
    hist_recons = [xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['basin','time']))[['swe_pred']].assign_coords(combo=f.split("/")[-1].split(".")[0]) for f in hist_files]
    hist_recons = xr.concat(hist_recons,dim='combo')
    hist_recons['time'] = hist_recons['time'].astype('datetime64[ns]')
    hist_recons = hist_recons.assign_coords(product=('combo',[s.split("_")[0] for s in hist_recons['combo'].values]))

    hist_trends = xr.apply_ufunc(theil_sen,hist_recons['swe_pred'],input_core_dims=[['time']],output_core_dims=[[]],vectorize=True)
    hist_trends.name = 'swe_trend'
    hist_trends.attrs = {"units":"$km^3/yr$","description":"Mann-Kendall trend in 1 April SWE, 1981-2020"}
    hist_trends = hist_trends.to_dataset()
    hist_trends['swe_trend_pct'] = 100*hist_trends['swe_trend']/hist_recons['swe_pred'].mean("time") # convert to pct of long-term average
    hist_trends['swe_trend_pct'].attrs = {"units":"%/yr","description":"Mann-Kendall trend in 1 April SWE, 1981-2020"}
    hist_trends.to_netcdf(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc'))
    
if not os.path.exists(os.path.join(project_dir,'data','basin_scale','noacc_recon_trends.nc')):
    # estimate counterfactual trends in parallel
    pool = Pool(16)
    res = pool.map(calc_trend_mp,noacc_files)
    res = xr.concat(res,dim='combo')
    pool.close()
    res.to_netcdf(os.path.join(project_dir,'data','basin_scale','noacc_recon_trends.nc'))

obs_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs_trends.nc')).rename({"swe":"swe_trend_pct"})
hist_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc')).sel(basin=obs_trends['basin'])
noacc_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','noacc_recon_trends.nc')).sel(basin=obs_trends['basin'])


# In[14]:


# get forced (historical minus counterfactual) response
def get_fr(combo):
    c_hist = hist_trends.sel(combo=combo)
    c_noacc = noacc_trends.where(noacc_trends['combo'].str.startswith(combo),drop=True)
    c_fr = c_hist['swe_trend_pct']-c_noacc
    c_fr = c_fr.drop("product")
    return c_fr

pool = Pool(8)
res = pool.map(get_fr,hist_trends['combo'].values)
forced_resp = xr.concat(res,dim='combo')
forced_resp = forced_resp.rename({"swe_noacc_trend_pct":"acc_effect",
                                  "swe_noacc_p_trend_pct":"acc_p_effect",
                                  "swe_noacc_t_trend_pct":"acc_t_effect"})

grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other','grdc_basins'))
basin_dict = dict(zip(grdc_basins['MRBID'],grdc_basins['RIVER_BASI']))

pool.close()


# In[30]:


# ensemble mean trends
recon_em = 10*hist_trends['swe_trend_pct'].mean("combo").to_dataframe() # get into percent per decade
recon_em = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(recon_em,left_on='MRBID',right_index=True)


obs_em = obs_trends['swe_trend_pct'].mean("product").to_dataframe()
obs_em = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(obs_em,left_on='MRBID',right_index=True)


# In[31]:


# fraction of products/reconstructions/obs-model combos that agree on sign of SWE trend
obs_agree = (np.sign(obs_trends['swe_trend_pct'])==-1).mean("product").to_dataframe()
recon_agree = (np.sign(hist_trends['swe_trend_pct'])==-1).mean("combo").to_dataframe()
fr_agree = (np.sign(forced_resp)==-1).mean("combo").to_dataframe()

obs_agree = obs_agree.rename(columns={"swe_trend_pct":"frac_agree"})
recon_agree = recon_agree.rename(columns={"swe_trend_pct":"frac_agree"})
fr_agree = fr_agree.rename(columns={"acc_effect":"frac_agree_all",
                            "acc_p_effect":"frac_agree_p",
                            "acc_t_effect":"frac_agree_t"})

obs_em = obs_em.merge(obs_agree,left_on='MRBID',right_index=True)
recon_em = recon_em.merge(recon_agree,left_on='MRBID',right_index=True)

forced_resp_df = forced_resp.mean("combo").to_dataframe()
forced_resp_df = 10*forced_resp_df # pct/decade
forced_resp_df = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(forced_resp_df,left_on='MRBID',right_index=True)
forced_resp_df = forced_resp_df.merge(fr_agree,left_on='MRBID',right_index=True)


# In[34]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='ticks',font_scale=2)

# create colormap
levels = np.arange(-10,5.1,1)
dec_levels = levels[:-5]
inc_levels = levels[-6:]
dec_cols = plt.get_cmap('RdBu')(np.linspace(0.05,0.45,len(dec_levels)))
inc_cols = plt.get_cmap("RdBu")(np.linspace(0.55,0.95,len(inc_levels)))
cols = np.concatenate([dec_cols,inc_cols])
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-10,vmax=5)



fig = plt.figure(figsize=(24,26))
gs = gridspec.GridSpec(nrows=3,ncols=2,wspace=0.05,hspace=0.0,height_ratios=[0.8,0.8,1.5],figure=fig)

ax1 = plt.subplot(gs[0,0],projection=ccrs.Miller())
obs_em[(obs_em['frac_agree']<=0.2)|(obs_em['frac_agree']>=0.8)].plot(ax=ax1,column='swe_trend_pct',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
obs_em[(obs_em['frac_agree']>0.2)&(obs_em['frac_agree']<0.8)].plot(ax=ax1,column='swe_trend_pct',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax1.title.set_text("OBSERVED")

ax2 = plt.subplot(gs[0,1],projection=ccrs.Miller())
recon_em[(recon_em['frac_agree']<=0.2)|(recon_em['frac_agree']>=0.8)].plot(ax=ax2,column='swe_trend_pct',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
recon_em[(recon_em['frac_agree']>0.2)&(recon_em['frac_agree']<0.8)].plot(ax=ax2,column='swe_trend_pct',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax2.title.set_text("RECONSTRUCTED")

ax4 = plt.subplot(gs[1,0],projection=ccrs.Miller())
forced_resp_df[(forced_resp_df['frac_agree_t']<=0.2)|(forced_resp_df['frac_agree_t']>=0.8)].plot(ax=ax4,column='acc_t_effect',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
forced_resp_df[(forced_resp_df['frac_agree_t']>0.2)&(forced_resp_df['frac_agree_t']<0.8)].plot(ax=ax4,column='acc_p_effect',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax4.title.set_text("FORCED TEMPERATURE EFFECT")

ax5 = plt.subplot(gs[1,1],projection=ccrs.Miller())
forced_resp_df[(forced_resp_df['frac_agree_p']<=0.2)|(forced_resp_df['frac_agree_p']>=0.8)].plot(ax=ax5,column='acc_p_effect',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
forced_resp_df[(forced_resp_df['frac_agree_p']>0.2)&(forced_resp_df['frac_agree_p']<0.8)].plot(ax=ax5,column='acc_p_effect',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax5.title.set_text("FORCED PRECIPITATION EFFECT")

ax6 = plt.subplot(gs[2,:],projection=ccrs.Miller())
forced_resp_df[(forced_resp_df['frac_agree_all']<=0.2)|(forced_resp_df['frac_agree_all']>=0.8)].plot(ax=ax6,column='acc_effect',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
forced_resp_df[(forced_resp_df['frac_agree_all']>0.2)&(forced_resp_df['frac_agree_all']<0.8)].plot(ax=ax6,column='acc_effect',hatch='//',cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax6.title.set_text("TOTAL FORCED EFFECT")


labels = ['a','b','c','d','e']
for i,ax in enumerate([ax1,ax2,ax4,ax5,ax6]):
    ax.set_aspect(1.346)
    ax.coastlines("10m")
    ax.text(0.01,0.975,labels[i],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)

sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm,ax=[ax1,ax2,ax4,ax5,ax6],orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.02)
cbar.ax.set_xlabel("MARCH SWE TREND (%/DECADE)",labelpad=15)
plt.savefig(os.path.join(project_dir,'figures','fig3.png'),bbox_inches='tight',dpi=400)
# plt.savefig(os.path.join(project_dir,'nature_figures','fig3.pdf'),bbox_inches='tight',dpi=400)


# In[ ]:




