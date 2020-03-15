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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/convection/"

# set up variable names and file name
varname = 'OMEGA'
varfname = "500hPa_omega"
varstr = r"$\omega 500$"
var_res = "fv09"
var_unit = r'$\times 10^{-3} Pa/s$'
fname1 = var_res+"_OMEGA_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_OMEGA_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_OMEGA_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_OMEGA_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_OMEGA_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# define pressure level
plevel = 500

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


# plot for all responses in one
def plotalldiff(lons, lats, res1, tt1, clim1, res2, tt2, clim2, res3, tt3, clim3, opt):
    fig = plt.figure()

    # total response
    ax1 = fig.add_subplot(311)
    # ax1.set_title(r'$\Delta_{total} \omega$'+str(plevel), fontsize=5, pad=3)
    ax1.set_title('a) Total response', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-10., 10.1, 1.)
    cs = map.contourf(x, y, res1, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
#    csc= ax1.contour(x, y, clim1, levels=np.arange(-120.,120.1,50.), linewidths=0.5, colors='k')
    if (opt == 1):
        levels = [0., 2.01, tt1.max()]
        csm = ax1.contourf(x, y, tt1, levels=levels, colors='none', hatches=["", "....."], alpha=0)

#    ax1.set_ylabel("Latitudes [degrees]",fontsize=5)
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    # fast response
    ax2 = fig.add_subplot(312)
    # ax2.set_title(r'$\Delta_{fast} \omega$'+str(plevel), fontsize=5, pad=3)
    ax2.set_title('b) Atmospheric-forced', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res2, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
#    csc= ax2.contour(x, y, clim2, levels=np.arange(-120.,120.1,50.), linewidths=0.5, colors='k')
    if (opt == 1):
        levels = [0., 2.01, tt2.max()]
        csm = ax2.contourf(x, y, tt2, levels=levels, colors='none', hatches=["", "....."], alpha=0)

#    ax2.set_ylabel("Latitudes [degrees]",fontsize=5)
    ax2.xaxis.set_tick_params(labelsize=5)
    ax2.yaxis.set_tick_params(labelsize=5)

    # slow response
    ax3 = fig.add_subplot(313)
    ax3.set_title('c) Ocean-mediated', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res3, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
#    csc= ax3.contour(x, y, clim3, levels=np.arange(-120.,120.1,50.), linewidths=0.5, colors='k')
    if (opt == 1):
        levels = [0., 2.01, tt3.max()]
        csm = ax3.contourf(x, y, tt3, levels=levels, colors='none', hatches=["", "....."], alpha=0)

#    ax3.set_ylabel("Latitudes [degrees]",fontsize=5)
#    ax3.set_xlabel('Longitudes [degrees]',fontsize=5)
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.yaxis.set_tick_params(labelsize=5)

    # add colorbar.
#    fig.subplots_adjust(right=0.7,hspace = 0.15)
    cbar_ax = fig.add_axes([0.69, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+' ['+var_unit+']', fontsize=6, labelpad=0.7)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_with_siglev_aerosolsinone.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_aerosolsinone.png", dpi=600, bbox_inches='tight')

    plt.suptitle("Aerosol Responses "+varstr+" changes", fontsize=10, y=0.95)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_with_siglev_aerosolsinone.pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_aerosolsinone.pdf", bbox_inches='tight')

    plt.close(fig)


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
levs = fdata1.variables['lev'][:]

################################################################################################
# S1-Plot contours
################################################################################################
# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

levi = np.abs(levs - plevel).argmin()
print(levs[levi])

# read the monthly data for a larger region
var1 = fdata1.variables[varname][(iniyear-1)*12: (endyear)*12, levi, latli:latui+1, lonli:lonui+1]
var2 = fdata2.variables[varname][(iniyear-1)*12: (endyear)*12, levi, latli:latui+1, lonli:lonui+1]
var3 = fdata3.variables[varname][(iniyear-1)*12: (endyear)*12, levi, latli:latui+1, lonli:lonui+1]
var4 = fdata4.variables[varname][(iniyear-1)*12: (endyear)*12, levi, latli:latui+1, lonli:lonui+1]
var5 = fdata5.variables[varname][(iniyear-1)*12: (endyear)*12, levi, latli:latui+1, lonli:lonui+1]

var1 = var1 * 1000
var2 = var2 * 1000
var3 = var3 * 1000
var4 = var4 * 1000
var5 = var5 * 1000

#################################################################################################
# select jja
var1jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
var2jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
var3jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
var4jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
var5jjats = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
print(var1jjats.shape)

for iyear in range(endyear-iniyear+1):
    var1jjats[iyear, :, :] = np.mean(var1[iyear*12+5:iyear*12+8, :, :], axis=0)
    var2jjats[iyear, :, :] = np.mean(var2[iyear*12+5:iyear*12+8, :, :], axis=0)
    var3jjats[iyear, :, :] = np.mean(var3[iyear*12+5:iyear*12+8, :, :], axis=0)
    var4jjats[iyear, :, :] = np.mean(var4[iyear*12+5:iyear*12+8, :, :], axis=0)
    var5jjats[iyear, :, :] = np.mean(var5[iyear*12+5:iyear*12+8, :, :], axis=0)

# jjaual means
var1jja = np.mean(var1jjats, axis=0)
var2jja = np.mean(var2jjats, axis=0)
var3jja = np.mean(var3jjats, axis=0)
var4jja = np.mean(var4jjats, axis=0)
var5jja = np.mean(var5jjats, axis=0)

# exp1
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-100., 105., 10.)
cs = map.contourf(x, y, var1jja, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case1.png")
plt.title(case1+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case1.pdf")

# exp2
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-100., 105., 10.)
cs = map.contourf(x, y, var2jja, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.title(case2+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case2.png")
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case2.pdf")

# exp3
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-100., 105., 10.)
cs = map.contourf(x, y, var3jja, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
# cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case3.png")
plt.title(case3+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case3.pdf")

# exp4
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-100., 105., 10.)
cs = map.contourf(x, y, var4jja, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case4.png")
plt.title(case4+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case4.pdf")

# exp5
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-100., 105., 10.)
cs = map.contourf(x, y, var5jja, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case5.png")
plt.title(case5+" JJA "+varstr, fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_case5.pdf")


#################################################################################################
# JJA means changes

# all forcings
vardiff, varttest = getstats(var2jjats, var1jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allforcings.png")
plt.title("All forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allforcings.pdf")

# GHG and natural forcing
vardiff, varttest = getstats(var5jjats, var1jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_GHGforcings.png")
plt.title("GHG and natural forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_GHGforcings.pdf")

# all aerosol forcing
vardiff, varttest = getstats(var2jjats, var5jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allaerosols.png")
plt.title("All Aerosol forcings JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_allaerosols.pdf")

# aerosol fast response1
vardiff, varttest = getstats(var2jjats, var4jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol1.png")
plt.title("Aerosol fast response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol1.pdf")

# aerosol fast response2
vardiff, varttest = getstats(var3jjats, var5jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol2.png")
plt.title("Aerosol fast response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol2.pdf")

# aerosol fast response check
vardiff, varttest = getstats(var2jjats-var4jjats, var3jjats-var5jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol_check.png")
plt.title("Aerosol fast response JJA "+varstr+" changes check", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_fastaerosol_check.pdf")

# aerosol slow response1
vardiff, varttest = getstats(var2jjats, var3jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol1.png")
plt.title("Aerosol slow response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol1.pdf")

# aerosol slow response2
vardiff, varttest = getstats(var4jjats, var5jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol2.png")
plt.title("Aerosol slow response JJA "+varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol2.pdf")

# aerosol slow response check
vardiff, varttest = getstats(var2jjats-var3jjats, var4jjats-var5jjats)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
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
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit)
# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol_check.png")
plt.title("Aerosol slow response JJA "+varstr+" changes check", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_slowaerosol_check.pdf")


################################################################################################
# S2-Plot in one graph
################################################################################################

plt.clf()
fig = plt.figure(4)

# all forcings
ax1 = fig.add_subplot(221)
ax1.set_title('All forcings', fontsize=6)
vardiff, varttest = getstats(var1jjats, var2jjats)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
clevs = np.arange(-10., 10.1, 1.)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)

# GHG and natural forcings
ax2 = fig.add_subplot(222)
ax2.set_title('GHG and Natural forcings', fontsize=6)
vardiff, varttest = getstats(var5jjats, var1jjats)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)

# Aerosol fast forcings
ax3 = fig.add_subplot(223)
ax3.set_title('Aerosol fast response', fontsize=6)
vardiff, varttest = getstats(var2jjats, var4jjats)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)

# Aerosol slow forcings
ax4 = fig.add_subplot(224)
ax4.set_title('Aerosol slow response', fontsize=6)
vardiff, varttest = getstats(var2jjats, var3jjats)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
              llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
map.drawcoastlines(linewidth=0.3)
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
cs = map.contourf(x, y, vardiff, clevs, cmap=cm.bwr, alpha=0.9, extend="both")
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x, y, varttest, levels=levels, colors='none', hatches=["", "..."], alpha=0)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cs, cax=cbar_ax)
cbar.ax.tick_params(labelsize=5)
cbar.set_label(var_unit, fontsize=5)

# add title
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_forcingsin1.png")
fig.suptitle(varstr+" changes", fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_forcingsin1.pdf")


# plot in one

res1, tt1 = getstats(var2jjats, var5jjats)
clim1 = np.mean(var5jjats, axis=0)

res2, tt2 = getstats((var2jjats+var3jjats)/2, (var4jjats+var5jjats)/2)
clim2 = np.mean((var4jjats+var5jjats)/2, axis=0)

res3, tt3 = getstats((var2jjats+var4jjats)/2, (var3jjats+var5jjats)/2)
clim3 = np.mean((var3jjats+var5jjats)/2, axis=0)

plotalldiff(lons, lats, res1, tt1, clim1, res2, tt2, clim2, res3, tt3, clim3, 0)
plotalldiff(lons, lats, res1, tt1, clim1, res2, tt2, clim2, res3, tt3, clim3, 1)
