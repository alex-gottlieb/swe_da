#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
project_data_dir = os.path.join(project_dir,'data')


# In[4]:


gl_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','gl_mask.nc')) # Greenland mask
swe_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','swe_mask.nc')) # SWE mask

# observed trends (incl. in situ)
obs_trends = xr.open_dataset(os.path.join(project_data_dir,'regrid_2deg','obs_trends.nc'))
obs_trends_em = obs_trends.mean('product') # mean of gridded products
obs_trends_em = obs_trends_em.assign_coords(product='Ens. Mean')
obs_trends = xr.concat([obs_trends,obs_trends_em],dim='product')
obs_trends = obs_trends.where((gl_mask['gl_mask']>0.9)&(swe_mask['swe']==1))


# In[6]:


obs_trends


# In[10]:


obs_agree['swe_trend_pct'].plot()


# In[12]:



sns.set(style='ticks',font_scale=1.8)
levels = np.arange(-2.5,2.6,0.5)
cols = plt.get_cmap('RdBu')(np.linspace(0,0.99,len(levels)+1))
cmap = mpl.colors.ListedColormap(cols[1:-1])
cmap.set_under(cols[0])
cmap.set_over(cols[-1])
norm = plt.Normalize(vmin=-2.5,vmax=2.5)

obs_agree = (np.sign(obs_trends['swe_trend_pct'])==1).drop_sel(product='Ens. Mean').mean("product")

fig = plt.figure(figsize=(20,14))
gs = gridspec.GridSpec(nrows=3,ncols=2,width_ratios=[1,1,],hspace=0.05,wspace=0.05,figure=fig)
axes = []

labels = ['a','b','c','d','e','f',]
for i,m in enumerate(obs_trends['product'].values):
    ax = plt.subplot(gs[i],projection=ccrs.Miller())
    obs_trends['swe_trend_pct'].sel(product=m).plot(ax=ax,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False)
    ax.title.set_text(m.split("_")[0])
    ax.coastlines("10m")
    ax.set_extent([-180,180,0,80],ccrs.PlateCarree())
    ax.text(0.01,0.975,labels[i],ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)
    if m == 'Ens. Mean':
        obs_agree.plot.contourf(ax=ax,colors='none',levels=[0,0.21,0.79,1],hatches=[None,'////',None],add_colorbar=False)
        ax.title.set_text("Ensemble Mean")
    axes.append(ax)
    
sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
sm.set_array([])
# cax = plt.subplot(gs[2,:])
cbar = fig.colorbar(sm,ax=axes,orientation="horizontal",drawedges=False,ticks=levels,extend='both',shrink=0.8,pad=0.03)
cbar.ax.set_xlabel("MARCH SWE TREND (%/DECADE)",labelpad=5)
cbar.ax.set_xticklabels((10*levels).astype(int))
plt.savefig(os.path.join(project_dir,'figures','figED1.png'),bbox_inches='tight',dpi=400)
# plt.savefig(os.path.join(root_dir,'agottlieb','swe_da','nature_figures','ed_fig1.jpg'),bbox_inches='tight',dpi=400)


# In[ ]:




