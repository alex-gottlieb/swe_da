#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xarray as xr
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
import re
import pandas as pd


# In[40]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')

swe_obs = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs.nc'))
swe_obs = swe_obs.where(swe_obs['swe'].min(['product','time'])>0,drop=True)
noacc_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','noacc_recon_trends.nc')).sel(basin=swe_obs['basin'])
hist_trends = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','hist_recon_trends.nc')).sel(basin=swe_obs['basin'])


# In[41]:


cmip_mods = list(set(["_".join(c.split("_")[-2:]) for c in noacc_trends['combo'].values]))
cmip_mods.sort()

cmip_mods
hist_trends_exp = []
for m in cmip_mods:
    ds_copy = hist_trends.copy()
    new_combos = [f'{c}_{m}' for c in hist_trends['combo'].values]
    ds_copy['combo'] = new_combos
    hist_trends_exp.append(ds_copy)
hist_trends_exp = xr.concat(hist_trends_exp,dim='combo')


combined = xr.merge([10*hist_trends_exp[['swe_trend_pct']],
                     10*noacc_trends,
                    ],
                    join='inner')
combined['acc_effect'] = combined['swe_trend_pct']-combined['swe_noacc_trend_pct']
combined['acc_p_effect'] = combined['swe_trend_pct']-combined['swe_noacc_p_trend_pct']
combined['acc_t_effect'] = combined['swe_trend_pct']-combined['swe_noacc_t_trend_pct']


# In[42]:


acc_df = combined[['acc_effect']].mean("combo").to_dataframe()


# In[43]:


grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other','grdc_basins'))
basin_ids = dict(zip(grdc_basins['RIVER_BASI'],grdc_basins['MRBID']))
basin_ids_rev = {v:k for k,v in basin_ids.items()}


# # Model Structure

# In[44]:


first_mem = [m for m in cmip_mods if m.endswith('01')]
mod_trends = []
for m in first_mem:
    m_trends = combined.where(combined['combo'].str.endswith(m),drop=True)
    m_trends['combo'] = ['_'.join(s.split("_")[:-2]) for s in m_trends['combo'].values]
    m_ens_mean = m_trends['acc_effect'].mean("combo")
    m_ens_mean = m_ens_mean.assign_coords(model=m)
    mod_trends.append(m_ens_mean)
mod_trends = xr.concat(mod_trends,dim='model')
mod_u = mod_trends.std("model")
mod_u = mod_u.where(mod_u!=0)
mod_u_df = mod_u.to_dataframe()
mod_u_df = mod_u_df.rename(columns={"acc_effect":"mod_u"})
# mod_u_df = grdc_basins[['MRBID','geometry']].merge(mod_u_df,left_on='MRBID',right_index=True)


# # Internal Variability

# In[45]:


le_trends = combined.where(combined['combo'].str.contains("MIROC6"),drop=True)
mem_nos = [int(c.split("_")[-1]) for c in le_trends['combo'].values]
le_trends = le_trends.assign_coords(i=(("combo"),mem_nos))


# In[46]:


mem_forced_trends = []
for i in range(50):
    _trends = le_trends.where(le_trends['i']==i,drop=True)
    _trends['combo'] = ['_'.join(s.split("_")[:-2]) for s in _trends['combo'].values]
    em_trend = _trends[['acc_effect','acc_p_effect','acc_t_effect']].mean('combo')
    em_trend = em_trend.assign_coords(i=i)
    mem_forced_trends.append(em_trend)
mem_forced_trends = xr.concat(mem_forced_trends,dim='i')
iv_u = mem_forced_trends.std("i")
iv_u_df = iv_u.to_dataframe()
iv_u_df = iv_u_df.rename(columns={"acc_effect":"iv_u",
                                  "acc_p_effect":"iv_p",
                                  "acc_t_effect":"iv_t"})
# iv_u_df = grdc_basins[['MRBID','geometry']].merge(iv_u_df,left_on='MRBID',right_index=True)


# # SWE Empirical Uncertainty

# In[47]:


products = ['ERA5-Land','JRA-55','MERRA-2','Snow-CCI','TerraClimate']
swe_trends = []
for p in products:
    p_trends = combined.where(combined['combo'].str.startswith(p),drop=True)
#     p_trends['combo'] = ['_'.join(s.split("_")[:-2]) for s in p_trends['combo'].values]
    p_ens_mean = (p_trends['acc_effect']).mean("combo")
    p_ens_mean = p_ens_mean.assign_coords(product=p)
    swe_trends.append(p_ens_mean)
swe_trends = xr.concat(swe_trends,dim='product')
swe_u = swe_trends.std("product")
swe_u = swe_u.where(swe_u!=0)
swe_u_df = swe_u.to_dataframe()
swe_u_df = swe_u_df.rename(columns={"acc_effect":"swe_u"})
# swe_u_df = grdc_basins[['MRBID','geometry']].merge(swe_u_df,left_on='MRBID',right_index=True)


# # T&P Empirical Uncertainty

# In[48]:


tp_combos = list(set(["_".join(c.split("_")[1:3]) for c in combined['combo'].values]))


# In[49]:


tp_trends = []
for c in tp_combos:
    c_trends = combined.where(combined['combo'].str.contains(c)&~combined['combo'].str.startswith(c),drop=True)
    c_ens_mean = c_trends['acc_effect'].mean("combo")
    c_ens_mean = c_ens_mean.assign_coords(tp_tps=c)
    tp_trends.append(c_ens_mean)
tp_trends = xr.concat(tp_trends,dim='tp_combo')
tp_u = tp_trends.std("tp_combo")
tp_u = tp_u.where(tp_u!=0)
tp_u_df = tp_u.to_dataframe()
tp_u_df = tp_u_df.rename(columns={"acc_effect":"tp_u"})
# tp_u_df = grdc_basins[['MRBID','geometry']].merge(tp_u_df,left_on='MRBID',right_index=True)


# In[50]:


all_u = mod_u_df.merge(iv_u_df,left_index=True,right_index=True).merge(swe_u_df,left_index=True,right_index=True).merge(tp_u_df,left_index=True,right_index=True)
all_u = all_u.dropna()


# In[51]:


f = all_u.sum(axis=1) / np.sqrt(np.power(all_u,2).sum(axis=1))
u_ci = pd.DataFrame(index=all_u.index,columns=[f"{c}_ci" for c in all_u.columns])
for c in all_u.columns:
    u_ci[f"{c}_ci"] = 1.654*all_u[c]/f


# In[52]:


all_u = all_u.drop(columns=['iv_t','iv_p'])


# In[64]:


sig_noise_df = all_u.copy()
for c in sig_noise_df.columns:
    sig_noise_df[c] = np.abs(acc_df['acc_effect'])>np.abs(sig_noise_df[c])
sig_noise_df.columns = [f"{c}_frac" for c in sig_noise_df.columns]


# In[53]:


total_u = all_u.sum(axis=1,min_count=3)
total_u.name = 'total_u'
ci_total_u = u_ci.sum(axis=1)


# In[54]:


frac_u = pd.DataFrame(index=all_u.index,columns=[f"{c}_frac" for c in all_u.columns])


# In[55]:


for c in all_u.columns:
    frac_u[f"{c}_frac"] = all_u[c]/total_u


# In[56]:


frac_u = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(frac_u,left_on='MRBID',right_index=True)


# In[58]:


frac_u['source_max'] = frac_u[frac_u.columns[3:]].idxmax(axis=1)


# In[59]:


max_cmap = {"swe_u_frac":'#984ea3',
            "tp_u_frac":'#ff7f00',
            "mod_u_frac":'#4daf4a',
            "iv_u_frac":'#e41a1c',
           }
frac_u['source_col'] = frac_u['source_max'].map(max_cmap)


# In[65]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
sns.set(style='ticks',font_scale=1.7)

levels = np.arange(0,0.51,0.05)
cols = plt.get_cmap('YlOrRd')(np.linspace(0,0.95,len(levels)))
cmap = mpl.colors.ListedColormap(cols[:-1])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=0,vmax=0.5)
titles = ['SWE OBSERVATIONS','TEMPERATURE & PRECIPITATION OBSERVATIONS','CLIMATE MODEL STRUCTURE','CLIMATE MODEL INTERNAL VARIABILITY']


labels = ['a','b','c','d','e','f',]

axes = []

fig = plt.figure(figsize=(24,30))
gs = gridspec.GridSpec(nrows=3,ncols=2,wspace=0.05,hspace=0.0,height_ratios=[1.5,0.8,0.8],figure=fig)

ax = plt.subplot(gs[0,:],projection=ccrs.Miller())
frac_u.plot(ax=ax,color=frac_u['source_col'],transform=ccrs.PlateCarree(),edgecolor='black',lw=0.5,legend=False)
ax.title.set_text("DOMINANT SOURCE OF UNCERTAINTY")
ax.coastlines("10m")
ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax.text(0.01,0.975,labels[0],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)

axes.append(ax)

from matplotlib.patches import Patch
leg_el = [Patch(facecolor=c,label=v) for v,c in max_cmap.items()]
leg = ax.legend(leg_el,['SWE OBS.','T&P OBS.','MODEL STUCTURE','MODEL IV'],ncol=4,loc='lower left',framealpha=1)

for i,c in enumerate(['swe_u_frac', 'tp_u_frac','mod_u_frac','iv_u_frac',]):
    hatch = sig_noise_df[sig_noise_df[c]==True].index
    no_hatch = sig_noise_df[sig_noise_df[c]==False].index
    ax = plt.subplot(gs[i+2],projection=ccrs.Miller())
    frac_u[frac_u['MRBID'].isin(hatch)].plot(ax=ax,column=c,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,hatch='\\\\',edgecolor='black',lw=0.5,legend=False)
    frac_u[frac_u['MRBID'].isin(no_hatch)].plot(ax=ax,column=c,transform=ccrs.PlateCarree(),cmap=cmap,norm=norm,edgecolor='black',lw=0.5,legend=False)
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.text(0.01,0.975,labels[i+1],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)
    ax.title.set_text(titles[i])
    axes.append(ax)
sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm,ax=axes,orientation="horizontal",drawedges=False,ticks=levels,extend='max',shrink=0.8,pad=0.02)
cbar.ax.set_xlabel("FRACTIONAL UNCERTAINTY IN FORCED SWE TREND (%)",labelpad=15)
cbar.ax.set_xticklabels((100*levels).astype(int))
plt.savefig(os.path.join(project_dir,'figures','ed_fig7.jpg'),bbox_inches='tight',dpi=400)
plt.show()


# In[ ]:




