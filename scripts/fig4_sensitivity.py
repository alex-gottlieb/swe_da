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
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
data_dir = os.path.join(project_dir,'data','regrid_0.5deg')
swe_dir = os.path.join(data_dir,'swe')
t_dir = os.path.join(data_dir,'tmean')


# In[3]:


# March SWE observational ensemble
swe_ens = xr.concat([xr.open_dataset(os.path.join(swe_dir,f)).sel(time=slice("1981","2020")).resample(time='AS-OCT').mean().assign_coords(product=f.split(".")[0]) for f in os.listdir(swe_dir)],dim='product')
swe_ens = swe_ens.drop_sel(product='InSitu')
swe_ens['swe_pct'] = 100*swe_ens['swe']/swe_ens['swe'].mean("time")

# NDJFM T observational ensemble
t_ens = xr.concat([xr.open_dataset(os.path.join(t_dir,f)).sel(time=slice("1980","2020")).assign_coords(product=f.split(".")[0]) for f in os.listdir(t_dir)],dim='product')
t_ens = t_ens.where((t_ens['time.month']>=11)|(t_ens['time.month']<=3)).resample(time='AS-OCT').mean()
t_ens = t_ens.sel(time=swe_ens['time'])


# In[4]:


swe_mask =  xr.open_dataset(os.path.join(data_dir,'swe_mask.nc')) # keep cells where >1/2 of all product years have non-zero March SWE
gl_mask = xr.open_dataset(os.path.join(data_dir,'gl_mask.nc'))

swe_ens = swe_ens.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))
t_ens = t_ens.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))


# In[5]:


def multiple_regression(*args):
    y = args[0]
    if np.isnan(y).sum()==len(y):
        return np.repeat(np.nan,len(args))
    else:
        y = np.nan_to_num(y)
 
    X = np.concatenate([x.reshape(-1,1) for x in args[1:]], axis=1)
    X = np.concatenate([np.ones(len(y)).reshape(-1,1),X],axis=1)
    try:
        mod = OLS(y,X).fit()
        return mod.params
    except:
        return np.repeat(np.nan,len(args))

# get dSWE/dT for all SWE-temperature data product combinations
all_sens = []
for swe_prod in swe_ens['product'].values:
    for t_prod in t_ens['product'].values:
        dswe_dt = xr.apply_ufunc(multiple_regression,
                                 swe_ens['swe'].sel(product=swe_prod),
                                 t_ens['tmean'].sel(product=t_prod),
                                 input_core_dims=[['time'],['time']],
                                 output_core_dims=[['beta']],
                                 vectorize=True)
        dswe_dt_pct = 100*dswe_dt/swe_ens['swe'].sel(product=swe_prod).mean("time")
        dswe_dt_pct = dswe_dt_pct.where(swe_ens['swe'].median("time").min('product')>0)
        dswe_dt_pct.name = 'dswe_dt'
        t_clim = t_ens['tmean'].sel(product=t_prod).mean("time")
        merged = xr.merge([dswe_dt_pct.isel(beta=1),t_clim],compat='override')
        merged = merged.assign_coords(combo=f"{swe_prod}_{t_prod}")
        all_sens.append(merged)
        print(swe_prod,t_prod)
        
all_obs_sens = xr.concat(all_sens,dim='combo').drop("product")

# get mean and std of sensitivity in rolling 5-degree window of climatological temperature
obs_df = all_obs_sens.to_dataframe().dropna()
obs_means = []
obs_stds = []
for t in np.arange(-37,4):
    _df = obs_df[(obs_df['tmean']>t-2.5)&(obs_df['tmean']<=t+2.5)]['dswe_dt']
    obs_means.append(_df.mean())
    obs_stds.append(_df.std())
obs_means = np.array(obs_means)
obs_stds = np.array(obs_stds)


# In[6]:


#get dSWE/dT for all basin-scale reconstructions
hist_dir = os.path.join(project_dir,'data','basin_scale','hist_recons')
hist_files = [os.path.join(hist_dir,f) for f in os.listdir(hist_dir)]
hist_files.sort()

keep_basins = xr.open_dataset(os.path.join(project_dir,'data','basin_scale','mar_swe_obs_trends.nc'))['basin']
recons = [xr.Dataset.from_dataframe(pd.read_csv(f).set_index(['basin','time'])).assign_coords(combo=f.split("/")[-1].split(".")[0]) for f in hist_files]
recons = xr.concat(recons,dim='combo').sel(basin=keep_basins)
recons['time'] = recons['time'].astype('datetime64[ns]')
recons = recons.assign_coords(product=('combo',[s.split("_")[0] for s in recons['combo'].values]))

basins = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','basins.nc'))
basin_t = t_ens.groupby(basins['basin']).mean().sel(basin=keep_basins)
basin_t_clim = basin_t.mean("time")

# get tmean corresponding to each basin-scale reconstruction
all_recon_t = xr.concat([basin_t.sel(product=f.split("_")[-1].split(".")[0]) for f in hist_files],dim='combo')

# combine SWE reconstruction and tmean
recon_merged = xr.merge([recons['swe_pred'].drop("product"),all_recon_t]).dropna("basin")

# calculate sensitivity
basin_t_sens = xr.apply_ufunc(multiple_regression,recon_merged['swe_pred'],recon_merged['tmean'],input_core_dims=[['time'],['time']],output_core_dims=[['beta']],vectorize=True)
basin_t_sens_pct = 100*basin_t_sens/recon_merged['swe_pred'].mean("time")
basin_t_sens_pct.name = 'dswe_dt'
merged_basin = xr.merge([basin_t_sens_pct.isel(beta=1),all_recon_t.mean('time')])

# get mean and std of sensitivity in rolling 5-degree window of climatological temperature
recon_sens_df = merged_basin.to_dataframe()
recon_means = []
recon_stds = []
for t in np.arange(-37,4):
    _df = recon_sens_df[(recon_sens_df['tmean']>t-2.5)&(recon_sens_df['tmean']<=t+2.5)]['dswe_dt']
    recon_means.append(_df.mean())
    recon_stds.append(_df.std())
recon_means = np.array(recon_means)
recon_stds = np.array(recon_stds)


# In[7]:


# get dSWE/dT at all in situ locations
in_situ = pd.read_csv(os.path.join(project_dir,'data','in_situ','all_in_situ.csv'))
keep_stns = (in_situ.groupby("station_id").count()['swe'] >= 35) & (in_situ.groupby("station_id").min()['swe']>0)
keep_stns = keep_stns[keep_stns].index
in_situ = in_situ[in_situ['station_id'].isin(keep_stns)]
t_cols = [c for c in in_situ.columns if 'tmean' in c]
in_situ['ndjfm_t'] = in_situ[t_cols].mean(axis=1)

def reg(x):
    try:
        return OLS.from_formula("swe~ndjfm_t",data=x).fit().params[1]
    except:
        return np.nan
    
# calculate sensitivity
dswe_dt = in_situ.groupby('station_id').apply(reg)
dswe_dt.name = 'dswe_dt'


t_clim = in_situ.groupby('station_id').agg({'ndjfm_t':"mean","swe":"mean"})
t_clim = t_clim.merge(dswe_dt,left_index=True,right_index=True)
t_clim['dswe_dt_pct'] = 100*t_clim['dswe_dt']/t_clim['swe'] # convert to percent of median

# get mean and std of sensitivity in rolling 5-degree window of climatological temperature
in_situ_means = []
in_situ_stds = []
for t in np.arange(-37,4):
    _df = t_clim[(t_clim['ndjfm_t']>t-2.5)&(t_clim['ndjfm_t']<=t+2.5)]['dswe_dt_pct']
    in_situ_means.append(_df.mean())
    in_situ_stds.append(_df.std())
in_situ_means = np.array(in_situ_means)
in_situ_stds = np.array(in_situ_stds)


# In[8]:


# get dSWE/dT for CMIP6 HIST simulations
mod_t_dir = os.path.join(project_dir,'data','regrid_2deg','tmean_hist')
t_mods = os.listdir(mod_t_dir)
mod_t_files = [os.path.join(mod_t_dir,f) for f in os.listdir(mod_t_dir)]
mod_t_files.sort()

mod_swe_dir = os.path.join(project_dir,'data','regrid_2deg','snw_hist')
swe_mods = os.listdir(mod_swe_dir)
mod_swe_files = [os.path.join(mod_swe_dir,f) for f in os.listdir(mod_swe_dir)]
mod_swe_files.sort()
shared_mods = list(set(t_mods)&set(swe_mods))
shared_mods.sort()

from datetime import datetime
def fix_time(ds):
    ds['time'] = [datetime(y,3,1) for y in ds['time.year'].values]
    return ds

mod_swe = xr.concat([fix_time(xr.open_dataset(os.path.join(mod_swe_dir,m))).assign_coords(model=m.split(".")[0]) for m in shared_mods],dim='model')
mod_t = []
for m in shared_mods:
    with xr.open_dataset(os.path.join(mod_t_dir,m)) as ds:
        ds_ndjfm = ds.where((ds['time.month']>=11)|(ds['time.month']<=3)).resample(time='AS-NOV').mean()
        ds_ndjfm['time'] = [datetime(y+1,3,1) for y in ds_ndjfm['time.year'].values]
        ds_ndjfm = ds_ndjfm.sel(time=slice("1981","2020"))
        ds_ndjfm = ds_ndjfm.assign_coords(model=m.split(".")[0])
        mod_t.append(ds_ndjfm)
mod_t = xr.concat(mod_t,dim='model',coords='minimal')
mod_merged = xr.merge([mod_t,mod_swe],join='inner')
mod_merged = mod_merged.where(mod_merged['snw'].min("time")>0)
mod_dswe_dt = xr.apply_ufunc(multiple_regression,mod_merged['snw'],mod_merged['tmean'],input_core_dims=[['time'],['time']],output_core_dims=[['beta']],vectorize=True)

mod_dswe_dt_pct = 100*mod_dswe_dt/mod_merged['snw'].mean("time")
mod_dswe_dt_pct.name = 'dswe_dt_pct'

mod_combined = xr.merge([mod_dswe_dt_pct.isel(beta=1),mod_t.mean("time")],join='inner')
df = mod_combined.to_dataframe().dropna()
mod_means = []
mod_stds = []
mod_counts = []
for t in np.arange(-37,4):
    _df = df[(df['tmean']>t-2.5)&(df['tmean']<=t+2.5)]['dswe_dt_pct']
#     _df = _df.clip(lower=-100)
    mod_means.append(_df.mean())
    mod_stds.append(_df.std())
mod_means = np.array(mod_means)
mod_stds = np.array(mod_stds)


# In[11]:


# get NH snow mass and population in each temperature bin for insets
swe_clim = swe_ens.mean("product").mean("time")
area = xr.open_dataset(os.path.join(project_dir,'data','regrid_0.5deg','area.nc'))
swe_clim['swe_gt'] = swe_clim['swe']*area['area']/1e12
swe_t_clim = xr.merge([swe_clim,t_ens.mean("product").mean("time")]).to_dataframe()
swe_t_clim['t_bin'] = pd.cut(swe_t_clim['tmean'],bins=np.arange(-40,6,2),labels=np.arange(-40,6,2)[:-1]+1)
swe_by_t = swe_t_clim.groupby("t_bin").agg({"swe_gt":"sum"})
swe_by_t['swe_gt_frac'] = swe_by_t['swe_gt']/swe_by_t['swe_gt'].sum()


pop_by_t = pd.read_csv(os.path.join(project_dir,'data','basin_scale','pop_by_tmean.csv'))


# In[14]:


import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='ticks',font_scale=2)
colors=sns.color_palette('deep',4)

fig=plt.figure(figsize=(32,10))
gs = gridspec.GridSpec(nrows=1,ncols=2,figure=fig)
ax = plt.subplot(gs[0])

# plot mean sensitivity, shade std
# gridded obs
ax.plot(np.arange(-37,4),obs_means,color=colors[0],lw=7)
ax.fill_between(np.arange(-37,4),obs_means-obs_stds,obs_means+obs_stds,color=colors[0],alpha=0.3)
# basin-scale reconstructions
ax.plot(np.arange(-37,4),recon_means,color=colors[1],lw=7)
ax.fill_between(np.arange(-37,4),recon_means-recon_stds,recon_means+recon_stds,color=colors[1],alpha=0.3)
# in situ
ax.plot(np.arange(-37,4),in_situ_means,color=colors[2],lw=7)
ax.fill_between(np.arange(-37,4),in_situ_means-in_situ_stds,in_situ_means+in_situ_stds,color=colors[2],alpha=0.3)
# climate models
ax.plot(np.arange(-37,4),mod_means,color=colors[3],lw=7)
ax.fill_between(np.arange(-37,4),mod_means-mod_stds,mod_means+mod_stds,color=colors[3],alpha=0.3)

ax.axhline(0,color='black',linestyle='--')
ax.set_xlabel("CLIMATOLOGICAL WINTER TEMPERATURE ($^{\circ}C$)")
ax.set_ylabel(r"$\frac{\partial SWE}{\partial T}$"+r"($\frac{\%}{^{\circ}C}$)",fontsize=30)
ax.set_yticks([-60,-40,-20,0,20])
ax.set_ylim(-125,30)
    
ax.text(-36,-30,"IN SITU OBSERVATIONS",ha='left',va='top',color=colors[2])
ax.text(-36,-39,"GRIDDED OBSERVATIONS",ha='left',va='top',color=colors[0])
ax.text(-36,-57,"BASIN-SCALE RECONSTRUCTIONS",ha='left',va='top',color=colors[1])
ax.text(-36,-48,"GRIDDED CLIMATE MODELS",ha='left',va='top',color=colors[3])

# snow mass by temperature inset
ax.bar(swe_by_t.index,height=150*swe_by_t['swe_gt_frac'],bottom=-105,width=2,color='white',edgecolor='black',lw=3)
ax.axhline(-85,color='black',)
ax.axvline(-8,color='red')
ax.text(-36.5,-87,'MARCH SWE DISTRIBUTION',ha='left',va='top',)
ax.text(-8.5,-87,'81%',ha='right',va='top',fontsize=16)
ax.text(-7.5,-87,'19%',ha='left',va='top',fontsize=16)

# population by temperature inset
ax.bar(pop_by_t['t_bin'],height=pop_by_t['pop']/1.5e8,bottom=-125,width=2,color='white',edgecolor='black',lw=3)
ax.axhline(-105,color='black')
ax.text(-36.5,-109,'POPULATION DISTRIBUTION',ha='left',va='top',)
ax.text(-8.5,-109,'570 MILLION',ha='right',va='top',fontsize=16)
ax.text(-7.5,-109,'2.1 BILLION',ha='left',va='top',fontsize=16)
ax.set_xlim(-37,3)
ax.text(0.01,0.99,'a',ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)

ax2 = plt.subplot(gs[1])
pop_plot_df = pd.read_csv(os.path.join(project_dir,'data','fig4b_data.csv'))

# colors for EOC SWE change
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
c = [swe_cmap(swe_norm(x)) for x in pop_plot_df['delta_swe_fut']]

# plot population vs. delta Q, sized by EOC T change, colored by EOC SWE change
g = ax2.scatter(x=pop_plot_df['log_pop'],y=pop_plot_df['delta_q_fut'],s=3**pop_plot_df['eoc_t'],color='none',edgecolors=c,lw=2)
ax2.axhline(0,color='black',linestyle='--')
# ax.axvline(0,color='black',linestyle='--')
ax2.set_xlabel("POPULATION")
ax2.set_ylabel("END-OF-CENTURY SWE-DRIVEN\nRUNOFF CHANGE (%)")

swe_sm = plt.cm.ScalarMappable(norm=swe_norm,cmap=swe_cmap)
swe_sm.set_array([])
swe_cbar = fig.colorbar(swe_sm,ax=ax2,orientation="vertical",drawedges=False,ticks=swe_levels,shrink=0.9,pad=0.02)
swe_cbar.ax.set_ylabel("END-OF-CENTURY SWE CHANGE (%)",labelpad=15)
ax2.set_ylim(-85,110)
ax2.set_yticks(np.linspace(-80,100,10))
ax2.set_xlim(1.5,9)
ticks = [[np.power(10,x),5*np.power(10,x)] for x in np.arange(2,9)]
ticks_flat = [x for sl in ticks for x in sl]
ax2.set_xticks(np.log10(ticks_flat))
ax2.set_xticklabels(['1 HUNDRED','','1 THOUSAND','','10 THOUSAND','','100 THOUSAND','','1 MILLION','','10 MILLION','','100 MILLION','',])
ax2.set_xticklabels(['1 HUNDRED','','','','10 THOUSAND','','','','1 MILLION','','','','100 MILLION','',])
ax2.scatter(np.linspace(2,5,6),np.repeat(-60,6),s=3**np.arange(2,8),color='w',edgecolor='black')
texts = [f'{x}'+'$^{\circ}$C' for x in np.arange(2,8)]
for i,x in enumerate(np.linspace(2,5,6)):
    ax2.text(x,-75,texts[i],ha='center',va='center')
ax2.text(3.5,-45,'END-OF-CENTURY WARMING',ha='center',va='center')
ax2.text(0.01,0.99,'b',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)

plt.savefig(os.path.join(project_dir,'figures','fig4.png'),bbox_inches='tight',dpi=400)

plt.show()


# In[ ]:




