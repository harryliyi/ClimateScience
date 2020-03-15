# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
plt.switch_backend('agg')


# set up data directories and filenames
case1 = "vrseasia_AMIP_1979_to_2005"
case2 = "ne30_ne30_AMIP_1979_to_2005"
case3 = "f19_f19_AMIP_1979_to_2005"
case4 = "f09_f09_AMIP_1979_to_2005"

fdir1 = "/scratch/d/dylan/harryli/cesm1/inputdata/seasia_30_x4/topo/"
fdir2 = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
fdir3 = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
fdir4 = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"

topofile1 = "topo_seasia_30_x4_smooth.nc"
topofile2 = "USGS-gtopo30_ne30np4_16xdel2-PFC-consistentSGH.nc"
topofile3 = "USGS-gtopo30_0.9x1.25_remap_c051027.nc"
topofile4 = "USGS-gtopo30_1.9x2.5_remap_c050602.nc"

# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/topos/'

# set up variable names and file name
varname = 'hgt'
var_longname = 'Topographic Height'
varstr = 'surface_height'
var_unit = 'm'


# define the contour plot region
latbounds = [0, 25]
lonbounds = [85, 125]
# latbounds = [-15, 25]
# lonbounds = [90, 145]


# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

############################################################################
# define functions
############################################################################

# data reader


def readtopo(varname, fdir, fname):

    dataset = nc.Dataset(fdir+fname)

    res = dataset.variables[varname][:]
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    res = res/9.8

    return res, lats, lons

############################################################################
# read data
############################################################################


print('read topographic information...')

varin = 'PHIS'
var1, lats1, lons1 = readtopo(varin, fdir1, topofile1)
var2, lats2, lons2 = readtopo(varin, fdir2, topofile2)
var3, lats3, lons3 = readtopo(varin, fdir3, topofile3)
var4, lats4, lons4 = readtopo(varin, fdir4, topofile4)

print(lats1)
print(lons1)

# plot the data

varmax = np.amax(var1)
varmin = np.amin(var1)
print(varmax)
print(varmin)

clevs = np.arange(varmin, 1600, 20)
colormap = cm.RdYlBu_r

plt.clf()
figsize = (8, 6)
fig = plt.figure(figsize=figsize)

############################################################################
# plot for vrcesm
ax = fig.add_subplot(2, 2, 1)
ax.set_title('a) '+cesm_legends[0], fontsize=7, pad=5)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()

# draw lat/lon lines
parallels = np.arange(latbounds[0], latbounds[1], 10)
meridians = np.arange(lonbounds[0], lonbounds[1], 10)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.02)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.02)

x, y = map(lons1, lats1)
# plot the contour
cs = map.contourf(x, y, var1, clevs, cmap=colormap, alpha=0.9, extend="both", tri=True)

# set x/y tick label size
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

############################################################################
# plot for ne30
ax = fig.add_subplot(2, 2, 2)
ax.set_title('b) '+cesm_legends[1], fontsize=7, pad=5)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()

# draw lat/lon lines
parallels = np.arange(latbounds[0], latbounds[1], 10)
meridians = np.arange(lonbounds[0], lonbounds[1], 10)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.02)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.02)

x, y = map(lons2, lats2)
# plot the contour
cs = map.contourf(x, y, var2, clevs, cmap=colormap, alpha=0.9, extend="both", tri=True)

# set x/y tick label size
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

############################################################################
# plot for fv0.9x1.25
ax = fig.add_subplot(2, 2, 3)
ax.set_title('c) '+cesm_legends[2], fontsize=7, pad=5)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()

# draw lat/lon lines
parallels = np.arange(latbounds[0], latbounds[1], 10)
meridians = np.arange(lonbounds[0], lonbounds[1], 10)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.02)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.02)

mlons, mlats = np.meshgrid(lons3, lats3)
x, y = map(mlons, mlats)
# plot the contour
cs = map.contourf(x, y, var3, clevs, cmap=colormap, alpha=0.9, extend="both")

# set x/y tick label size
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

############################################################################
# plot for fv1.9x2.5
ax = fig.add_subplot(2, 2, 4)
ax.set_title('d) '+cesm_legends[3], fontsize=7, pad=5)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.5)
map.drawcountries()

# draw lat/lon lines
parallels = np.arange(latbounds[0], latbounds[1], 10)
meridians = np.arange(lonbounds[0], lonbounds[1], 10)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.02)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.02)

mlons, mlats = np.meshgrid(lons4, lats4)
x, y = map(mlons, mlats)
# plot the contour
cs = map.contourf(x, y, var4, clevs, cmap=colormap, alpha=0.9, extend="both")

# set x/y tick label size
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

# add color bar at bottom
fig.subplots_adjust(bottom=0.22, wspace=0.15, hspace=0.15)
cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=4)
cbar.set_label(var_longname+' ['+var_unit+']', fontsize=6, labelpad=-0.2)

# save figure
plt.savefig(outdir+'vrcesm_SEA_topographs.png', bbox_inches='tight', dpi=600)
plt.suptitle('Representations of Topography in CESMs', fontsize=7, y=0.95)
plt.savefig(outdir+'vrcesm_SEA_topographs.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
