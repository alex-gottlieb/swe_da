#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from cartopy.mpl import geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import seaborn as sns
import re
from datetime import datetime
import geopandas as gpd
import pymannkendall as mk


# In[2]:


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
project_dir = os.path.join(root_dir,'agottlieb','git_repos','swe_da')
project_data_dir = os.path.join(project_dir,'data')


# In[3]:


swe_dir = os.path.join(project_data_dir,'regrid_0.5deg','swe')
swe_files = [os.path.join(swe_dir,f) for f in os.listdir(swe_dir) if 'InSitu' not in f]
swe_files.sort()


# In[7]:


swe_ens = xr.concat([xr.open_dataset(f).resample(time='AS-OCT').mean() for f in swe_files],dim='product') # make sure dates exactly the same
swe_ens = swe_ens.sel(time=slice("1980","2019")) # WY starts in previous year
swe_mask =  xr.open_dataset(os.path.join(project_data_dir,'regrid_0.5deg','swe_mask.nc')) # keep cells where >1/2 of all product years have non-zero March SWE
gl_mask = xr.open_dataset(os.path.join(project_data_dir,'regrid_0.5deg','gl_mask.nc'))

def calc_grid_cell_areas(ds, lat_name="lat", lon_name="lon", r_earth=6.371e6):
    """
    Calculate the area (in km^2) of DataSet or DataArray grid cells, using the following formula:
    A = r^2 * (lon1 - lon0) * (sin(lat1) - sin(lat0))
    where r is the radius of the earth, lon1 and lon0 are the max and min bounds of the grid cell along the x-axis, lat1 and lat0
    along the y-axis.
    Note: currently assumes coordinates refer to center of grid cells (but not necessarily that they're evenly spaced)
    """
    def get_bounds(coords_1d):
        """
        Calculates boundaries of grid cells in one dimension
        """
        diffs = np.diff(coords_1d)
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])
        min_bounds = coords_1d - diffs[:-1] / 2
        max_bounds = coords_1d + diffs[1:] / 2
        return np.array([min_bounds, max_bounds]).transpose()
    # get boundaries of grid cells, convert to radians
    lat_bounds = get_bounds(ds[lat_name])
    lat_bounds_rad = np.deg2rad(lat_bounds)
    lon_bounds = get_bounds(ds[lon_name])
    lon_bounds_rad = np.deg2rad(lon_bounds)
    # get widths and heights (in radians) of grid cells
    y_lens = np.sin(lat_bounds_rad[:, 1]) - np.sin(lat_bounds_rad[:, 0])
    x_lens = lon_bounds_rad[:, 1] - lon_bounds_rad[:, 0]
    # calculate areas in km^2 given widths and heights in radians
    areas = (r_earth ** 2) * np.outer(y_lens, x_lens)
    area_da = xr.DataArray(np.abs(areas), coords=[ds[lat_name],ds[lon_name]], dims=["lat", "lon"])
    return area_da

# get grid-cell areas for calculating snow mass
area_da = calc_grid_cell_areas(gl_mask)

hem_swe_ens = []
for f in swe_files:
    with xr.open_dataset(f) as ds:
        ds = ds.where((gl_mask['gl_mask']==1)&(swe_mask==1)) # apply masks
        ds['time'] = [datetime(y,3,1) for y in ds['time.year'].values]
        hem_swe = (ds['swe']*area_da).sum(['lat','lon'])/1e12 # snowmass in km^3
        hem_swe = hem_swe.assign_coords(product=f.split("/")[-1].split(".")[0])
        hem_swe_ens.append(hem_swe)
        print(f)
hem_swe_ens = xr.concat(hem_swe_ens,dim='product')
hem_swe_ens.name = 'swe'
hem_swe_ens = hem_swe_ens.to_dataset()
hem_swe_ens = hem_swe_ens.sel(time=slice("1981","2020"))


# In[10]:


basin_swe = xr.open_dataset(os.path.join(project_data_dir,'basin_scale','mar_swe_obs.nc'))
basin_swe = basin_swe.where(basin_swe['swe'].min(['product','time'])>0,drop=True)


# In[14]:


t_dir = os.path.join(project_data_dir,'regrid_0.5deg','tmean')
t_prods = ['BEST.nc','ERA5-Land.nc','MERRA-2.nc','CPC.nc']
weights = np.cos(np.deg2rad(gl_mask.lat)) # weights proportional to grid-cell size
hem_t_ens = []
basin_t_ens = []
basins = xr.open_dataset(os.path.join(project_data_dir,'regrid_0.5deg','basins.nc'))
for f in t_prods:
    with xr.open_dataset(os.path.join(t_dir,f)) as ds:
        ds = ds.sel(time=ds['time'][(ds['time.month']>=11)|(ds['time.month']<=3)]).resample(time='AS-NOV').mean().sel(time=slice("1980","2019")) # get NDJFM average
        ds = ds.where((gl_mask['gl_mask']==1)&(swe_mask==1)) # apply masks
        hem_t = ds['tmean'].weighted(weights).mean(['lat','lon']) # weight by area
        hem_t = hem_t.assign_coords(product=f.split(".")[0])
        hem_t_ens.append(hem_t)
        
        basin_t = ds['tmean'].groupby(basins['basin']).mean() # basin-scale average 
        basin_t = basin_t.assign_coords(product=f.split(".")[0])
        basin_t_ens.append(basin_t)
hem_t_ens = xr.concat(hem_t_ens,dim='product')
hem_t_ens.name = 'tmean'
hem_t_ens = hem_t_ens.to_dataset()

basin_t_ens = xr.concat(basin_t_ens,dim='product')
basin_t_ens.name = 'tmean'
basin_t_ens = basin_t_ens.to_dataset()
basin_t_ens = basin_t_ens.sel(basin=basin_swe['basin'])
basin_t_ens.to_netcdf(os.path.join(project_data_dir,'basin_scale','ndjfm_tmean_obs.nc'))


# In[20]:


basin_pop = xr.open_dataset(os.path.join(project_data_dir,'basin_scale','population.nc'))

grdc_basins = gpd.read_file(os.path.join(project_data_dir,'grdc_basins'))
basin_ids = dict(zip(grdc_basins['RIVER_BASI'],grdc_basins['MRBID']))
basin_ids_rev = {v:k for k,v in basin_ids.items()}

basin_pop_df = basin_pop.to_dataframe()
grdc_basins = grdc_basins.merge(basin_pop_df,left_on='MRBID',right_index=True,how='left')
q_basins = [4219, 4238, 4231, 4405, 6202, 6903, 6204, 6259, 2309, 2434, 2436, 2320] # basins w/ significant snowfall-runoff ratio
most_pop = grdc_basins[grdc_basins.MRBID.isin(q_basins)].set_index("MRBID").loc[q_basins].reset_index()


# In[22]:


con_pop = grdc_basins.groupby("CONTINENT").agg({"pop":"sum"}) # continent-scale population
con_dict = dict(zip(grdc_basins['MRBID'],grdc_basins['CONTINENT']))

basin_swe = basin_swe.assign(dict(continent=(('basin'),[con_dict[b] for b in basin_swe['basin'].values])))
con_swe = basin_swe.groupby("continent").sum()

basin_t_ens = basin_t_ens.assign(dict(continent=(('basin'),[con_dict[b] for b in basin_t_ens['basin'].values])))
con_t = basin_t_ens.groupby("continent").mean()


# In[23]:


def theil_sen(ts):
    ts = ts[~np.isnan(ts)]
    try:
        out=mk.original_test(ts)
        return out.slope, out.intercept
    except:
        return np.nan, np.nan

# hemispheric trends
hem_swe_trends,_ = xr.apply_ufunc(theil_sen,hem_swe_ens['swe'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
hem_swe_trends = 1000*hem_swe_trends/hem_swe_ens['swe'].mean("time") # percent per decade

hem_t_trends,_ = xr.apply_ufunc(theil_sen,hem_t_ens['tmean'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
hem_t_trends = 10*hem_t_trends # degrees per decade

#basin-scale trends
basin_swe_trends, _ = xr.apply_ufunc(theil_sen,basin_swe['swe'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
basin_swe_trends = 1000*basin_swe_trends/basin_swe['swe'].mean("time") # percent per decade

basin_t_trends, _ = xr.apply_ufunc(theil_sen,basin_t_ens['tmean'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
basin_t_trends = 10*basin_t_trends # degrees per decade

# continental-scale trends
con_swe_trends,_ = xr.apply_ufunc(theil_sen,con_swe['swe'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
con_swe_trends = 1000*con_swe_trends/con_swe['swe'].mean("time") # percent per decade

con_t_trends,_ = xr.apply_ufunc(theil_sen,con_t['tmean'],input_core_dims=[['time']],output_core_dims=[[],[]],vectorize=True)
con_t_trends = 10*con_t_trends # degrees per decade


# In[24]:


# number of products that agree on sign of trend
n_agree_swe = (np.sign(basin_swe_trends)==1).sum("product").to_dataframe()
n_agree_swe = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(n_agree_swe,left_on='MRBID',right_index=True)

n_agree_t = (np.sign(basin_t_trends)==1).sum("product").to_dataframe()
n_agree_t = grdc_basins[['MRBID','RIVER_BASI','geometry']].merge(n_agree_t,left_on='MRBID',right_index=True)


# In[25]:


colors = ['#af8dc3','lightgrey','#7fbf7b',]
cmap = ListedColormap(colors)

sns.set(style='ticks',font_scale=1.5)
mpl.rcParams['pdf.fonttype'] = 42

fig = plt.figure(figsize=(24,20))
gs = gridspec.GridSpec(nrows=3,ncols=6,height_ratios=[0.35,0.03,0.4],wspace=0.2,hspace=0.15,figure=fig)
labels = ['c','d','e','f','g','h','i','j','k','l','m','n']
ax = plt.subplot(gs[0,:3],projection=ccrs.Miller())
n_agree_t[n_agree_t['tmean']==4].plot(ax=ax,transform=ccrs.PlateCarree(),color='#7fbf7b',edgecolor='black',lw=0.5,legend=False) # all agree increase
n_agree_t[n_agree_t['tmean']==0].plot(ax=ax,transform=ccrs.PlateCarree(),color='#af8dc3',edgecolor='black',lw=0.5,legend=False) # all agree decrease
n_agree_t[(n_agree_t['tmean']>0)&(n_agree_t['tmean']<4)].plot(ax=ax,transform=ccrs.PlateCarree(),color='lightgrey',edgecolor='black',lw=0.5,legend=False) # at least 1 disagree
ax.coastlines("10m")
ax.set_extent([-180,180,-10,80],ccrs.PlateCarree())
ax.set_aspect(1.346)
ax.title.set_text("WINTER TEMPERATURE")
ax.text(0.01,0.975,'a',ha='left',va='top',transform=ax.transAxes,weight='bold',fontsize=24,)
ax.text(0.01,0.05,"0%",ha='left',va='bottom',transform=ax.transAxes,color='#af8dc3',weight='bold',fontsize=20,)
ax.text(0.07,0.05,"10%",ha='left',va='bottom',transform=ax.transAxes,color='grey',weight='bold',fontsize=20,)
ax.text(0.15,0.05,"90%",ha='left',va='bottom',transform=ax.transAxes,color='#7fbf7b',weight='bold',fontsize=20,)

# plot ID numbers on major basins for subplots
xoff = [0,0,-0.5,-0.5,
        0,0,0,0,
        0,0,0,0]
yoff = [-5,-6,-2,-3.5,
        -5.5,-8.5,-6.8,-7,
        -1,-3,-4,-2.5]
for i, g in enumerate(most_pop['geometry']):
    x,y = g.centroid.xy # centoid of basin
    ax.text(x[0],y[0],str(i+1),ha='center',va='center',fontsize=14,fontweight='bold',transform=ccrs.PlateCarree()) # add number to plot


# inset for hemispheric-scale trends
ax_ins = inset_axes(ax,width="25%",height='20%',
                   loc='lower center',
                    bbox_to_anchor=(-0.1,0.1,1,1),
                    bbox_transform=ax.transAxes)
ax_ins.scatter(hem_t_trends.values,np.repeat(0.5,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
ax_ins.set_xlim(-1,1)
ax_ins.set_xticks(np.arange(-1,1.1,0.5))
ax_ins.title.set_text("$^{\circ}C$/DECADE,\n1981-2020")
ax_ins.set_ylim(0,1)
ax_ins.set_yticks([])
ax_ins.axvline(0,color='black',linestyle='--')
ax_ins.tick_params(labelsize=14)
ax_ins.patch.set_facecolor("white")
for tl in ax_ins.get_xticklabels():
    tl.set_backgroundcolor('white')

ax2 = plt.subplot(gs[0,3:],projection=ccrs.Miller())
n_agree_swe[n_agree_swe['swe']==5].plot(ax=ax2,transform=ccrs.PlateCarree(),color='#7fbf7b',edgecolor='black',lw=0.5,legend=False) # all agree increase
n_agree_swe[n_agree_swe['swe']==0].plot(ax=ax2,transform=ccrs.PlateCarree(),color='#af8dc3',edgecolor='black',lw=0.5,legend=False) # all agree decrease
n_agree_swe[(n_agree_swe['swe']>0)&(n_agree_swe['swe']<5)].plot(ax=ax2,transform=ccrs.PlateCarree(),color='lightgrey',edgecolor='black',lw=0.5,legend=False) # at least 1 disagree
ax2.coastlines("10m")
ax2.set_extent([-180,180,-10,80],ccrs.PlateCarree())
# ax2.set_aspect(np.pi/(18/8))
ax2.title.set_text("MARCH SNOWPACK")
ax2.text(0.01,0.975,'b',ha='left',va='top',transform=ax2.transAxes,weight='bold',fontsize=24,)
ax2.text(0.01,0.05,"26%",ha='left',va='bottom',transform=ax2.transAxes,color='#af8dc3',weight='bold',fontsize=20,)
ax2.text(0.09,0.05,"67%",ha='left',va='bottom',transform=ax2.transAxes,color='grey',weight='bold',fontsize=20,)
ax2.text(0.17,0.05,"7%",ha='left',va='bottom',transform=ax2.transAxes,color='#7fbf7b',weight='bold',fontsize=20,)

ax2.set_aspect(1.346)

# inset for hemispheric scale-trends
ax2_ins = inset_axes(ax2,width="25%",height='20%',
                   loc='lower center',
                    bbox_to_anchor=(-0.1,0.1,1,1),
                    bbox_transform=ax2.transAxes)
ax2_ins.scatter(hem_swe_trends.values,np.repeat(0.5,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
ax2_ins.set_xlim(-2.3,2.3)
ax2_ins.set_xticks(np.arange(-2,2.1,))
ax2_ins.title.set_text("%/DECADE,\n1981-2020")
ax2_ins.set_ylim(0,1)
ax2_ins.set_yticks([])
ax2_ins.axvline(0,color='black',linestyle='--')
ax2_ins.tick_params(labelsize=14)
ax2_ins.patch.set_facecolor("white")
for tl in ax2_ins.get_xticklabels():
    tl.set_backgroundcolor('white')
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm,ax=[ax,ax2],orientation='horizontal',ticks=[1/6,3/6,5/6],drawedges=False,shrink=0.7,pad=0.05)
cbar.ax.set_xticklabels(['AGREEMENT ON\nDECREASING TREND','PRODUCTS\nDISAGREE','AGREEMENT ON\nINCREASING TREND'])

# North American basins
na_ax = plt.subplot(gs[2,:2])
na_ax2 = na_ax.twiny()
for idx,row in most_pop.iloc[:4].iterrows():
    t = basin_t_trends.sel(basin=row['MRBID']).values
    swe = basin_swe_trends.sel(basin=row['MRBID']).values
    na_ax2.scatter(t,np.repeat(4.6-idx,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
    na_ax.scatter(swe,np.repeat(4.25-idx,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
    na_ax.text(-24,4.95-idx,re.sub(r' \([^)]*\)', '', row['RIVER_BASI']) + f" ({idx+1})",
                ha='left',va='top',
                fontweight='bold',
                zorder=4)
    na_ax.text(24,4.95-idx,f"{np.rint(row['pop']/1e6).astype(int)} MM",ha='right',va='top')
idx += 1
# continental-scale trends
na_t = con_t_trends.sel(continent='North America, Central America and the Caribbean').values
na_swe = con_swe_trends.sel(continent='North America, Central America and the Caribbean').values
na_ax2.scatter(na_t,np.repeat(4.6-idx,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
na_ax.scatter(na_swe,np.repeat(4.25-idx,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
na_ax.text(-24,4.95-idx,'CONTINENTAL',
            ha='left',va='top',
            fontweight='bold',
            zorder=4)
na_ax.text(24,4.95-idx,f"{np.rint(con_pop.loc['North America, Central America and the Caribbean','pop']/1e6).astype(int)} MM",ha='right',va='top')

# prettify axes  
na_ax2.set_xlim(-1,1)
na_ax.set_xlim(-25,25)
na_ax.hlines(np.arange(1,5,1),-25,25,color='black',lw=5)
na_ax.axvline(0,color='black',linestyle='--',zorder=1)
na_ax.set_xticks(np.arange(-25,26,5))
na_ax2.set_xticks(np.arange(-1,1.1,0.5))
na_ax.set_ylim(0,5)
na_ax.set_yticks([])
na_ax.set_xlabel("MARCH SWE TREND (%/DECADE)",color='cornflowerblue')
na_ax2.set_xlabel("WINTER T TREND ($^{\circ}C$/DECADE)",color='red')               
na_ax.spines['bottom'].set_color('cornflowerblue')
na_ax2.spines['bottom'].set_color('cornflowerblue')
na_ax.spines['top'].set_color('red')
na_ax2.spines['top'].set_color('red')
na_ax.xaxis.label.set_color('cornflowerblue')
na_ax2.xaxis.label.set_color('red')
na_ax.tick_params(axis='x', colors='cornflowerblue')
na_ax2.tick_params(axis='x', colors='red')
na_ax.text(0,5.6,'c. NORTH AMERICA',ha='center',va='bottom',fontsize=24,fontweight='bold')
    
# European basins
eur_ax = plt.subplot(gs[2,2:4])
eur_ax2 = eur_ax.twiny()
for idx,row in most_pop.iloc[4:8].iterrows():
    t = basin_t_trends.sel(basin=row['MRBID']).values
    swe = basin_swe_trends.sel(basin=row['MRBID']).values
    eur_ax2.scatter(t,np.repeat(4.6-idx+4,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
    eur_ax.scatter(swe,np.repeat(4.25-idx+4,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
    eur_ax.text(-24,4.95-idx+4,re.sub(r' \([^)]*\)', '', row['RIVER_BASI']) + f" ({idx+1})",
                ha='left',va='top',
                fontweight='bold',
                zorder=4)
    eur_ax.text(24,4.95-idx+4,f"{np.rint(row['pop']/1e6).astype(int)} MM",ha='right',va='top')

idx += 1
eur_t = con_t_trends.sel(continent='Europe').values
eur_swe = con_swe_trends.sel(continent='Europe').values
eur_ax2.scatter(eur_t,np.repeat(4.6-idx+4,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
eur_ax.scatter(eur_swe,np.repeat(4.25-idx+4,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
eur_ax.text(-24,4.95-idx+4,'CONTINENTAL',
            ha='left',va='top',
            fontweight='bold',
            zorder=4)
eur_ax.text(24,4.95-idx+4,f"{np.rint(con_pop.loc['Europe','pop']/1e6).astype(int)} MM",ha='right',va='top')

eur_ax2.set_xlim(-1,1)
eur_ax.set_xlim(-25,25)
eur_ax.hlines(np.arange(1,5,1),-25,25,color='black',lw=5)
eur_ax.axvline(0,color='black',linestyle='--',zorder=1)
eur_ax.set_xticks(np.arange(-25,26,5))
eur_ax2.set_xticks(np.arange(-1,1.1,0.5))
eur_ax.set_ylim(0,5)
eur_ax.set_yticks([])
eur_ax.set_xlabel("MARCH SWE TREND (%/DECADE)",color='cornflowerblue')
eur_ax2.set_xlabel("WINTER T TREND ($^{\circ}C$/DECADE)",color='red')               
eur_ax.spines['bottom'].set_color('cornflowerblue')
eur_ax2.spines['bottom'].set_color('cornflowerblue')
eur_ax.spines['top'].set_color('red')
eur_ax2.spines['top'].set_color('red')
eur_ax.xaxis.label.set_color('cornflowerblue')
eur_ax2.xaxis.label.set_color('red')
eur_ax.tick_params(axis='x', colors='cornflowerblue')
eur_ax2.tick_params(axis='x', colors='red')
eur_ax.text(0,5.6,'d. EUROPE',ha='center',va='bottom',fontsize=24,fontweight='bold')

# Asian basins
asia_ax = plt.subplot(gs[2,4:])
asia_ax2 = asia_ax.twiny()
for idx,row in most_pop.iloc[8:].iterrows():
    t = basin_t_trends.sel(basin=row['MRBID']).values
    swe = basin_swe_trends.sel(basin=row['MRBID']).values
    asia_ax2.scatter(t,np.repeat(4.6-idx+8,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
    asia_ax.scatter(swe,np.repeat(4.25-idx+8,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
    asia_ax.text(-24,4.95-idx+8,re.sub(r' \([^)]*\)', '', row['RIVER_BASI']) + f" ({idx+1})",
                ha='left',va='top',
                fontweight='bold',
                zorder=4)
    asia_ax.text(24,4.95-idx+8,f"{np.rint(row['pop']/1e6).astype(int)} MM",ha='right',va='top')
idx += 1
asia_t = con_t_trends.sel(continent='Asia').values
asia_swe = con_swe_trends.sel(continent='Asia').values
asia_ax2.scatter(asia_t,np.repeat(4.6-idx+8,4),color='red',marker='^',edgecolors='black',s=500,zorder=3)
asia_ax.scatter(asia_swe,np.repeat(4.25-idx+8,5),color='cornflowerblue',marker='s',edgecolors='black',s=500,zorder=3)
asia_ax.text(-24,4.95-idx+8,'CONTINENTAL',
            ha='left',va='top',
            fontweight='bold',
            zorder=4)
asia_ax.text(24,4.95-idx+8,f"{np.rint(con_pop.loc['Asia','pop']/1e6).astype(int)} MM",ha='right',va='top')

asia_ax2.set_xlim(-1,1)
asia_ax.set_xlim(-25,25)
asia_ax.hlines(np.arange(1,5,1),-25,25,color='black',lw=5)
asia_ax.axvline(0,color='black',linestyle='--',zorder=1)
asia_ax.set_xticks(np.arange(-25,26,5))
asia_ax2.set_xticks(np.arange(-1,1.1,0.5))
asia_ax.set_ylim(0,5)
asia_ax.set_yticks([])
asia_ax.set_xlabel("MARCH SWE TREND (%/DECADE)",color='cornflowerblue')
asia_ax2.set_xlabel("WINTER T TREND ($^{\circ}C$/DECADE)",color='red')               
asia_ax.spines['bottom'].set_color('cornflowerblue')
asia_ax2.spines['bottom'].set_color('cornflowerblue')
asia_ax.spines['top'].set_color('red')
asia_ax2.spines['top'].set_color('red')
asia_ax.xaxis.label.set_color('cornflowerblue')
asia_ax2.xaxis.label.set_color('red')
asia_ax.tick_params(axis='x', colors='cornflowerblue')
asia_ax2.tick_params(axis='x', colors='red')
asia_ax.text(0,5.6,'e. ASIA',ha='center',va='bottom',fontsize=24,fontweight='bold')

plt.savefig(os.path.join(project_dir,'figures','fig1.png'),bbox_inches='tight',dpi=400)


# In[ ]:




