#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
project_data_dir = os.path.join(project_dir,'data')


# In[23]:


gl_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','gl_mask.nc')) # Greenland mask
swe_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','swe_mask.nc')) # SWE mask

# CMIP6 HIST trends
hist_dir = os.path.join(project_data_dir,'regrid_2deg','hist_trends')
mods = ['ACCESS-CM2_r1i1p1f1',
        'ACCESS-ESM1-5_r1i1p1f1',
        'BCC-CSM2-MR_r1i1p1f1',
        'CanESM5_r1i1p1f1',
        'CNRM-CM6-1_r1i1p1f2',
        'GFDL-CM4_r1i1p1f1',
        'GFDL-ESM4_r1i1p1f1',
        'IPSL-CM6A-LR_r1i1p1f1',
        'MIROC6_r1i1p1f1',
        'MRI-ESM2-0_r1i1p1f1',
        'NorESM2-LM_r1i1p1f1']
hist_files = [os.path.join(hist_dir,f"{m}.nc") for m in mods]

hist_ens = xr.concat([xr.open_dataset(f).assign_coords(model=f.split("/")[-1].split("_")[0]) for f in hist_files],dim='model')
hist_ens = hist_ens.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))
hist_agree = (np.sign(hist_ens['snw_trend'])==1).mean("model")


# In[26]:


sns.set(style='ticks',font_scale=1.8)
levels = np.arange(-2.5,2.6,0.5)
cols = plt.get_cmap('RdBu')(np.linspace(0,0.99,len(levels)+1))
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-2.5,vmax=2.5)

fig = plt.figure(figsize=(20,28))
gs = gridspec.GridSpec(nrows=6,ncols=2,width_ratios=[1,1,],hspace=0.05,wspace=0.05,figure=fig)
axes = []

labels = ['a','b','c','d','e',
          'f','g','h','i','j','k','l']
for i,m in enumerate(hist_ens['model'].values):
    ax = plt.subplot(gs[i],projection=ccrs.Miller())
    hist_ens['snw_trend_pct'].sel(model=m).plot(ax=ax,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
    ax.title.set_text(m.split("_")[0])
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.text(0.01,0.975,labels[i],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)

    axes.append(ax)
    
ax = plt.subplot(gs[i+1],projection=ccrs.Miller())
hist_ens['snw_trend_pct'].mean("model").plot(ax=ax,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
ax.coastlines("10m")
ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
ax.text(0.01,0.975,labels[i+1],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)
hist_agree.plot.contourf(ax=ax,transform=ccrs.PlateCarree(),colors='none',levels=[0,0.21,0.79,1],hatches=[None,'////',None],add_colorbar=False)
ax.title.set_text("Ensemble Mean")

axes.append(ax)
sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
# cax = plt.subplot(gs[2,:])
cbar = fig.colorbar(sm,ax=axes,orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.01)
cbar.ax.set_xlabel("MARCH SWE TREND (%/DECADE)",labelpad=5)
cbar.ax.set_xticklabels((10*levels).astype(int))
plt.savefig(os.path.join(project_dir,'figures','figED2.png'),bbox_inches='tight',dpi=400)
# plt.savefig(os.path.join(root_dir,'agottlieb','swe_da','nature_figures','ed_fig2.jpg'),bbox_inches='tight',dpi=400)


# In[ ]:




