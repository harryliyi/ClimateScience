# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

#import libraries
import math as math
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case1 = "vrseasia_19501959_OBS"
case2 = "vrseasia_20002010_OBS"
case3 = "vrseasia_20002009_OBS_SUBAERSST_CESM1CAM5_SST"
case4 = "vrseasia_20002009_OBS_AEREMIS1950"
case5 = "vrseasia_20002009_OBS_AEREMIS1950_SUBAERSST_CESM1CAM5_SST"

expdir1 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case1+"/atm/"
expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case2+"/atm/"
expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case3+"/atm/"
expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case4+"/atm/"
expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case5+"/atm/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/circulation/"

# set up variable names and file name
variname = 'U850'
varjname = 'V850'
varfname = "850hPa_wind"
varstr = "850hPa wind"
var_unit = 'm/s'
var_res = "fv09"

fname1 = var_res+"_WIND_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_WIND_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_WIND_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_WIND_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_WIND_"+case5+".cam.h0.0001-0050.nc"

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

################################################################################################
# S0-Define functions
################################################################################################


def getstats(var1, var2):
    n1 = var1.shape[0]
    n2 = var2.shape[0]

    var1mean = np.mean(var1, axis=0)
    var2mean = np.mean(var2, axis=0)
    var1std = np.std(var1, axis=0)
    var2std = np.std(var2, axis=0)

    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff, abs(varttest)


################################################################################################
# S1-open climatological data
################################################################################################
# open data file
fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]

################################################################################################
# S1-Plot contours
################################################################################################
# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

# read the monthly data for a larger region
vari1 = fdata1.variables[variname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
vari2 = fdata2.variables[variname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
vari3 = fdata3.variables[variname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
vari4 = fdata4.variables[variname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
vari5 = fdata5.variables[variname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]

varj1 = fdata1.variables[varjname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
varj2 = fdata2.variables[varjname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
varj3 = fdata3.variables[varjname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
varj4 = fdata4.variables[varjname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
varj5 = fdata5.variables[varjname][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]

#################################################################################################
# select jja
vari1jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
vari2jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
vari3jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
vari4jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
vari5jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))

varj1jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
varj2jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
varj3jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
varj4jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
varj5jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))

print(vari1jjats.shape)

for iyear in range(endyear-iniyear+1):
    vari1jjats[iyear, :, :] = np.mean(vari1[iyear*12+5:iyear*12+8, :, :], axis=0)
    vari2jjats[iyear, :, :] = np.mean(vari2[iyear*12+5:iyear*12+8, :, :], axis=0)
    vari3jjats[iyear, :, :] = np.mean(vari3[iyear*12+5:iyear*12+8, :, :], axis=0)
    vari4jjats[iyear, :, :] = np.mean(vari4[iyear*12+5:iyear*12+8, :, :], axis=0)
    vari5jjats[iyear, :, :] = np.mean(vari5[iyear*12+5:iyear*12+8, :, :], axis=0)

    varj1jjats[iyear, :, :] = np.mean(varj1[iyear*12+5:iyear*12+8, :, :], axis=0)
    varj2jjats[iyear, :, :] = np.mean(varj2[iyear*12+5:iyear*12+8, :, :], axis=0)
    varj3jjats[iyear, :, :] = np.mean(varj3[iyear*12+5:iyear*12+8, :, :], axis=0)
    varj4jjats[iyear, :, :] = np.mean(varj4[iyear*12+5:iyear*12+8, :, :], axis=0)
    varj5jjats[iyear, :, :] = np.mean(varj5[iyear*12+5:iyear*12+8, :, :], axis=0)

# jjaual means
vari1jja = np.mean(vari1jjats, axis=0)
vari2jja = np.mean(vari2jjats, axis=0)
vari3jja = np.mean(vari3jjats, axis=0)
vari4jja = np.mean(vari4jjats, axis=0)
vari5jja = np.mean(vari5jjats, axis=0)

varj1jja = np.mean(varj1jjats, axis=0)
varj2jja = np.mean(varj2jjats, axis=0)
varj3jja = np.mean(varj3jjats, axis=0)
varj4jja = np.mean(varj4jjats, axis=0)
varj5jja = np.mean(varj5jjats, axis=0)

# exp1
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-36., 37., 2.)
cs = map.contourf(x, y, vari1jja, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
cq = map.quiver(x[::3, ::3], y[::3, ::3], vari1jja[::3, ::3], varj1jja[::3, ::3], scale=2., scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.suptitle(case1+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case1.pdf")

# exp2
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-36., 37., 2.)
cs = map.contourf(x, y, vari2jja, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
cq = map.quiver(x[::3, ::3], y[::3, ::3], vari1jja[::3, ::3], varj1jja[::3, ::3], scale=2., scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.suptitle(case2+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case2.pdf")

# exp3
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-36., 37., 2.)
cs = map.contourf(x, y, vari3jja, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
cq = map.quiver(x[::3, ::3], y[::3, ::3], vari1jja[::3, ::3], varj1jja[::3, ::3], scale=2., scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.suptitle(case3+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case3.pdf")

# exp4
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-36., 37., 2.)
cs = map.contourf(x, y, vari4jja, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
cq = map.quiver(x[::3, ::3], y[::3, ::3], vari1jja[::3, ::3], varj1jja[::3, ::3], scale=2., scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.suptitle(case4+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case4.pdf")

# exp5
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-36., 37., 2.)
cs = map.contourf(x, y, vari5jja, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
cq = map.quiver(x[::3, ::3], y[::3, ::3], vari1jja[::3, ::3], varj1jja[::3, ::3], scale=2., scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.suptitle(case5+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case5.pdf")


#################################################################################################
# JJA means changes

# all forcings
vardiff, varttest = getstats(vari2jjats, vari1jjats)
varjdiff = varj2jja - varj1jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("All forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allforcings.pdf")

# GHG and natural forcing
vardiff, varttest = getstats(vari5jjats, vari1jjats)
varjdiff = varj5jja - varj1jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("GHG and natural forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_GHGforcings.pdf")

# all aerosol forcing
vardiff, varttest = getstats(vari2jjats, vari5jjats)
varjdiff = varj2jja - varj5jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("All Aerosol forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allaerosols.pdf")

# aerosol fast response1
vardiff, varttest = getstats(vari2jjats, vari4jjats)
varjdiff = varj2jja - varj4jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol fast response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol1.pdf")

# aerosol fast response2
vardiff, varttest = getstats(vari3jjats, vari5jjats)
varjdiff = varj3jja - varj5jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol fast response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol2.pdf")

# aerosol fast response check
vardiff, varttest = getstats(vari2jjats-vari4jjats, vari3jjats-vari5jjats)
varjdiff = (varj2jja - varj4jja)-(varj3jja - varj5jja)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol fast response JJA "+varstr+" changes check", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol_check.pdf")

# aerosol slow response1
vardiff, varttest = getstats(vari2jjats, vari3jjats)
varjdiff = varj2jja - varj3jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol slow response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol1.pdf")

# aerosol slow response2
vardiff, varttest = getstats(vari4jjats, vari5jjats)
varjdiff = varj4jja - varj5jja

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol slow response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol2.pdf")

# aerosol slow response check
vardiff, varttest = getstats(vari2jjats-vari3jjats, vari4jjats-vari5jjats)
varjdiff = (varj2jja - varj3jja) - (varj4jja - varj5jja)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
# print(mlons)
# print(lons[lonli:lonui+1])
# print(x)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax.quiverkey(cq, 0.9, 0.9, 5, '5 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.suptitle("Aerosol slow response JJA "+varstr+" changes check", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol_check.pdf")


################################################################################################
# S2-Plot in one graph
################################################################################################

plt.clf()
fig = plt.figure(4)

# all forcings
ax1 = fig.add_subplot(221)
ax1.set_title('All forcings', fontsize=6)
vardiff, varttest = getstats(vari2jjats, vari1jjats)
varjdiff = varj2jja - varj1jja
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
clevs = np.arange(-4.5, 4.6, 0.5)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax1.quiverkey(cq, 1., 1., 5, '5 '+var_unit, labelpos='E', coordinates='axes', fontproperties={'size': 6})

# GHG and natural forcings
ax2 = fig.add_subplot(222)
ax2.set_title('GHG and Natural forcings', fontsize=6)
vardiff, varttest = getstats(vari5jjats, vari1jjats)
varjdiff = varj5jja - varj1jja
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax2.quiverkey(cq, 1., 1., 5, '5 '+var_unit, labelpos='E', coordinates='axes', fontproperties={'size': 6})

# Aerosol fast forcings
ax3 = fig.add_subplot(223)
ax3.set_title('Aerosol fast response', fontsize=6)
vardiff, varttest = getstats(vari2jjats, vari4jjats)
varjdiff = varj2jja - varj4jja
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax3.quiverkey(cq, 1., 1., 5, '5 '+var_unit, labelpos='E', coordinates='axes', fontproperties={'size': 6})

# Aerosol slow forcings
ax4 = fig.add_subplot(224)
ax4.set_title('Aerosol slow response', fontsize=6)
vardiff, varttest = getstats(vari2jjats, vari3jjats)
varjdiff = varj2jja - varj3jja
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
#csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
cq = map.quiver(x[::3, ::3], y[::3, ::3], vardiff[::3, ::3], varjdiff[::3, ::3], scale=0.25, scale_units='xy')
qk = ax4.quiverkey(cq, 1., 1., 5, '5 '+var_unit, labelpos='E', coordinates='axes', fontproperties={'size': 6})

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cs, cax=cbar_ax)
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit, fontsize=5)

# add title
fig.suptitle(varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_forcingsin1.pdf")
