#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xarray as xr
import numpy as np
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import seaborn as sns


# In[2]:


def list_files(d):
    files =[os.path.join(d,f) for f in os.listdir(d)]
    files.sort()
    return files


# In[4]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
project_data_dir = os.path.join(project_dir,'data')

pic_trend_dir = os.path.join(project_data_dir,'regrid_2deg','pic_trends')
histnat_trend_dir = os.path.join(project_data_dir,'regrid_2deg','hist-nat_trends')
hist_trend_dir = os.path.join(project_data_dir,'regrid_2deg','hist_trends')
pic_trend_files = list_files(pic_trend_dir)
histnat_trend_files = list_files(histnat_trend_dir)
hist_trend_files = list_files(hist_trend_dir)


# In[20]:


gl_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','gl_mask.nc')) # Greenland mask
swe_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','swe_mask.nc')) # SWE mask

# observed trends (incl. in situ)
obs_trends = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','obs_trends.nc'))
obs_trends_em = obs_trends.mean('product') # mean of gridded products
obs_trends_em = obs_trends_em.assign_coords(product='Ens. Mean')
obs_trends = xr.concat([obs_trends,obs_trends_em],dim='product')
obs_trends = obs_trends.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))

insitu_trends = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','insitu_trends.nc')).assign_coords(product='In situ')
obs_trends = xr.concat([obs_trends['swe_trend_pct'],insitu_trends['swe_trend_pct']],dim='product')


# In[24]:


# CMIP6 HIST trends
hist_trends = [xr.open_dataset(f) for f in hist_trend_files]
hist_mods = [f.split("/")[-1].split(".")[0] for f in hist_trend_files]
hist_trends = xr.concat(hist_trends,dim='model')
hist_trends['model'] = hist_mods
hist_trends = hist_trends.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))

# CMIP6 HIST-NAT trends
histnat_mods = [f.split("/")[-1].split(".")[0] for f in histnat_trend_files]
histnat_trends = [xr.open_dataset(f) for f in histnat_trend_files]
histnat_trends = xr.concat(histnat_trends,dim='model')
histnat_trends['model'] = histnat_mods
histnat_trends = histnat_trends.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))


# In[27]:


# only use models participating in all experiments
hist_mods = list(set([m.split("_")[0] for m in hist_mods]))
histnat_mods = list(set([m.split("_")[0] for m in histnat_mods]))
pic_mods = [f.split("/")[-1].split("_")[0] for f in pic_trend_files]

shared_mods = list(set(hist_mods)&set(histnat_mods)&set(pic_mods))
shared_mods.sort()


# In[30]:


def corr(x1,x2): # function for calculating Spearman correlation for use with xr.apply_ufunc
    nas = np.logical_or(np.isnan(x1), np.isnan(x2))
    try:
        return spearmanr(x1[~nas],x2[~nas])[0]
    except:
        return np.nan
    
all_hist_pic = []
for m in shared_mods:
    # subset HIST trends from each model
    m_hist = hist_trends.sel(model=hist_trends['model'][hist_trends['model'].str.startswith(m)]) 
    
    # load PIC trends for model (different file name for 2 models)
    if m == 'IPSL-CM6A-LR':
        m_pic = xr.open_dataset(os.path.join(pic_trend_dir,f'{m}_r1i2p1f1.nc'))
    elif m == 'CNRM-CM6-1':
        m_pic = xr.open_dataset(os.path.join(pic_trend_dir,f'{m}_r1i1p1f2.nc'))
    else:
        m_pic = xr.open_dataset(os.path.join(pic_trend_dir,f'{m}_r1i1p1f1.nc'))
        
    # calculate pattern correlations between HIST trend and all unique 40-year trends from PIC
    hist_pic_corrs = xr.apply_ufunc(corr,m_hist['snw_trend_pct'],m_pic['snw_trend_pct'],input_core_dims=[['lat','lon'],['lat','lon']],vectorize=True)

    all_hist_pic.append(hist_pic_corrs)
    print(m)
    
# create 1-D array of all correlations
all_hist_pic_vals = np.concatenate([x.values.flatten() for x in all_hist_pic])


# In[39]:


# use first available ensemble member from each model
first = ['ACCESS-CM2_r1i1p1f1',
        'ACCESS-ESM1-5_r1i1p1f1',
        'BCC-CSM2-MR_r1i1p1f1',
        'CNRM-CM6-1_r1i1p1f2',
        'GFDL-CM4_r1i1p1f1',
        'GFDL-ESM4_r1i1p1f1',
        'IPSL-CM6A-LR_r1i1p1f1',
        'MIROC6_r1i1p1f1',
        'MRI-ESM2-0_r1i1p1f1',
        'NorESM2-LM_r1i1p1f1']

# pattern correlations between HIST/HIST-NAT and obs
hist_obs_corr = xr.apply_ufunc(corr,hist_trends['snw_trend_pct'].sel(model=first).mean("model"),obs_trends,input_core_dims=[['lat','lon'],['lat','lon']],vectorize=True)
histnat_obs_corr = xr.apply_ufunc(corr,histnat_trends['snw_trend_pct'].sel(model=first).mean("model"),obs_trends,input_core_dims=[['lat','lon'],['lat','lon']],vectorize=True)


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style='ticks',font_scale=1.8)
mpl.rcParams['pdf.fonttype'] = 42

levels = np.arange(-2.5,2.6,0.5)
cols = plt.get_cmap('RdBu')(np.linspace(0,0.99,len(levels)+1))
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-2.5,vmax=2.5)

product_order = ['ERA5-Land', 'JRA-55', 'MERRA-2', 'Snow-CCI', 'TerraClimate','Ens. Mean','In situ']
hist_obs_corr = hist_obs_corr.sel(product=product_order)
histnat_obs_corr = histnat_obs_corr.sel(product=product_order)

fig = plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(nrows=4,ncols=2,width_ratios=[1,1,],height_ratios=[1,1,0.02,1.75],wspace=0.02,hspace=0.02,figure=fig)
axes = []

ax1 = plt.subplot(gs[0,0],projection=ccrs.Miller())
obs_trends.sel(product='In situ').plot(ax=ax1,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
ax1.title.set_text('IN SITU')
# ax1.text(0.01,0.01,'a',ha='left',va='bottom',transform=ax1.transAxes,weight='bold',fontsize=24,)
ax2 = plt.subplot(gs[0,1],projection=ccrs.Miller())
obs_trends.sel(product='Ens. Mean').plot(ax=ax2,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
ax2.title.set_text('GRIDDED PRODUCTS')
# ax2.text(0.01,0.01,'b',ha='left',va='bottom',transform=ax2.transAxes,weight='bold',fontsize=24,)

ax3 = plt.subplot(gs[1,0],projection=ccrs.Miller())
hist_trends['snw_trend_pct'].sel(model=first).mean("model").plot(ax=ax3,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
ax3.title.set_text('CMIP6 HISTORICAL')
# ax3.text(0.01,0.01,'c',ha='left',va='bottom',transform=ax3.transAxes,weight='bold',fontsize=24,)

ax4 = plt.subplot(gs[1,1],projection=ccrs.Miller())
histnat_trends['snw_trend_pct'].sel(model=first).mean("model").plot(ax=ax4,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
ax4.title.set_text('CMIP6 HIST-NAT')
# ax4.text(0.01,0.01,'d',ha='left',va='bottom',transform=ax4.transAxes,weight='bold',fontsize=24,)

labels = ['a','b','c','d']
for i,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.text(0.01,0.975,labels[i],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)


sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
# cax = plt.subplot(gs[2,:])
cbar = fig.colorbar(sm,ax=[ax1,ax2,ax3,ax4],orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.05)
cbar.ax.set_xlabel("MARCH SWE TREND (%/DECADE)",labelpad=5)
cbar.ax.set_xticklabels((10*levels).astype(int))

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','black']
widths = 5*[5]+2*[12]
ax5 = plt.subplot(gs[3,:])
ax5.hist(all_hist_pic_vals.reshape(-1,1),
         weights=(np.ones_like(all_hist_pic_vals)/len(all_hist_pic_vals)).reshape(-1,1) # make sum to 1
         ,color='grey',alpha=0.3,edgecolor='black',
         bins=np.arange(-0.525,0.51,0.05))
ymin,ymax=ax5.get_ylim()

ax5.axvline(np.quantile(all_hist_pic_vals,0.95),color='orange',linestyle='dotted',lw=4)
ax5.text(np.quantile(all_hist_pic_vals,0.95)+0.01,0.127,'95th percentile',rotation=-90,color='orange',va='top')
ax5.axvline(np.quantile(all_hist_pic_vals,0.99),color='red',linestyle='dotted',lw=4)
ax5.text(np.quantile(all_hist_pic_vals,0.99)+0.01,0.127,'99th percentile',rotation=-90,color='red',va='top')

ax5.set_ylim(ymin,ymax)

# ax5.axvline(0,color='black',linestyle='--')
ax5.set_xlabel("SPATIAL CORRELATION WITH CMIP6 ENS. MEAN")
ax5.set_ylabel("DENSITY")
ax5.text(0.01,0.99,'e',ha='left',va='top',transform=ax5.transAxes,weight='bold',fontsize=24,)

markers = ['o','s','^','P','D','X','*']


for i,p in enumerate(product_order):
    ax5.scatter(histnat_obs_corr.sel(product=p),0.008,color='cornflowerblue',marker=markers[i],edgecolor='black',s=500)
    ax5.scatter(hist_obs_corr.sel(product=p),0.018,color='firebrick',marker=markers[i],edgecolor='black',s=500)

obs_labels = ['ERA5-Land', 'JRA-55', 'MERRA-2', 'Snow-CCI', 'TerraClimate','Ens. Mean','In situ']

# handles = [Patch(facecolor='grey',alpha=0.3),Line2D([0],[0],color='red',lw=3,linestyle='dotted'),Patch(facecolor='white')]+[Line2D([0],[0],color=c,lw=w) for c,w in zip(colors,widths)]
obs_handles = [Line2D([0], [0], marker=markers[i], color='w',markerfacecolor='black', markersize=20) for i in range(7)]
leg = ax5.legend(obs_handles,obs_labels,loc='upper right',title='OBSERVATIONAL\nPRODUCT')
plt.setp(leg.get_title(), multialignment='center')

ax5.set_xlim(-0.5,0.66)
ax5.set_xticks(np.linspace(-0.4,0.4,5))
ax5.text(0.65,0.028,r'$\rho$(HIST,PIC)',color='grey',ha='right',va='center',fontsize=24)
ax5.text(0.65,0.018,r'$\rho$(HIST,OBS)',color='firebrick',ha='right',va='center',fontsize=24)
ax5.text(0.65,0.008,r'$\rho$(HIST-NAT,OBS)',color='cornflowerblue',ha='right',va='center',fontsize=24)
ax5.axhline(y=0.023,xmin=(hist_obs_corr.min("product").values+0.5)/1.16,xmax=(hist_obs_corr.max("product").values+0.5)/1.16,color='firebrick',lw=5)
ax5.axhline(y=0.013,xmin=(histnat_obs_corr.min("product").values+0.5)/1.16,xmax=(histnat_obs_corr.max("product").values+0.5)/1.16,color='cornflowerblue',lw=5)

plt.savefig(os.path.join(project_dir,'figures','fig2.png'),bbox_inches='tight',dpi=400)
# plt.savefig(os.path.join(root_dir,'agottlieb','swe_da','nature_figures','fig2.pdf'),bbox_inches='tight',dpi=400)


# In[ ]:




