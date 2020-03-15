#This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

#S1-plot climatological data
#S2-plot contours  
#
#by Harry Li

#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import math as math

#set up data directories and filenames
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

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/climatology/"

#set up variable names and file name
varname = 'PRECT'
varstr  = "Total Precip"
var_res = "fv02"
fname1 = var_res+"_PREC_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_PREC_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_PREC_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_PREC_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_PREC_"+case5+".cam.h0.0001-0050.nc"

#define inital year and end year
iniyear = 1
endyear = 50

#define the contour plot region
latbounds = [ -15 , 45 ]
lonbounds = [ 60 , 160 ]

#define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

#month series
month = np.arange(1,13,1)
mname = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

################################################################################################
#S0-Define functions
################################################################################################
def getstats(var1,var2):
    n1=var1.shape[0]
    n2=var2.shape[0]

    var1mean = np.mean(var1,axis = 0)
    var2mean = np.mean(var2,axis = 0)
    var1std  = np.std(var1,axis = 0)
    var2std  = np.std(var2,axis = 0)

    vardiff  = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff,abs(varttest)

################################################################################################
#S1-open climatological data
################################################################################################
#open data file
fdata1  = Dataset(expdir1+fname1)
fdata2  = Dataset(expdir2+fname2)
fdata3  = Dataset(expdir3+fname3)
fdata4  = Dataset(expdir4+fname4)
fdata5  = Dataset(expdir5+fname5)

#read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]

################################################################################################
#S1-Plot contours
################################################################################################
# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

#read the monthly data for a larger region
var1 = fdata1.variables[varname][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
var2 = fdata2.variables[varname][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
var3 = fdata3.variables[varname][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
var4 = fdata4.variables[varname][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
var5 = fdata5.variables[varname][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]

var1 = var1 * 86400 * 1000
var2 = var2 * 86400 * 1000
var3 = var3 * 86400 * 1000
var4 = var4 * 86400 * 1000
var5 = var5 * 86400 * 1000

#################################################################################################
#select mam
var1mamts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var2mamts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var3mamts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var4mamts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var5mamts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
print(var1mamts.shape)

for iyear in range(endyear-iniyear+1):
    var1mamts[iyear,:,:] = np.mean(var1[iyear*12+5:iyear*12+8,:,:],axis=0)
    var2mamts[iyear,:,:] = np.mean(var2[iyear*12+5:iyear*12+8,:,:],axis=0)
    var3mamts[iyear,:,:] = np.mean(var3[iyear*12+5:iyear*12+8,:,:],axis=0)
    var4mamts[iyear,:,:] = np.mean(var4[iyear*12+5:iyear*12+8,:,:],axis=0)
    var5mamts[iyear,:,:] = np.mean(var5[iyear*12+5:iyear*12+8,:,:],axis=0)

#mamual means
var1mam = np.mean(var1mamts,axis=0)
var2mam = np.mean(var2mamts,axis=0)
var3mam = np.mean(var3mamts,axis=0)
var4mam = np.mean(var4mamts,axis=0)
var5mam = np.mean(var5mamts,axis=0)

#exp1
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
cs = map.contourf(x,y,var1mam,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case1+" MAM "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_case1.pdf")

#exp2
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
cs = map.contourf(x,y,var2mam,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case2+" MAM "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_case2.pdf")

#exp3
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
cs = map.contourf(x,y,var3mam,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case3+" MAM "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_case3.pdf")

#exp4
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
cs = map.contourf(x,y,var4mam,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case4+" MAM "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_case4.pdf")

#exp5
plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
cs = map.contourf(x,y,var5mam,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case5+" MAM "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_case5.pdf")


#################################################################################################
#MAM means changes

#all forcings
vardiff,varttest = getstats(var1mamts,var2mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("All forcings MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_allforcings.pdf")

#GHG and natural forcing
vardiff,varttest = getstats(var5mamts,var1mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("GHG and natural forcings MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_GHGforcings.pdf")

#all aerosol forcing
vardiff,varttest = getstats(var2mamts,var5mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("All Aerosol forcings MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_allaerosols.pdf")

#aerosol fast response1
vardiff,varttest = getstats(var2mamts,var4mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_fastaerosol1.pdf")

#aerosol fast response2
vardiff,varttest = getstats(var3mamts,var5mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_fastaerosol2.pdf")

#aerosol fast response check
vardiff,varttest = getstats(var2mamts-var4mamts,var3mamts-var5mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response MAM "+varstr+" changes check",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_fastaerosol_check.pdf")

#aerosol slow response1
vardiff,varttest = getstats(var2mamts,var3mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_slowaerosol1.pdf")

#aerosol slow response2
vardiff,varttest = getstats(var4mamts,var5mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response MAM "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_slowaerosol2.pdf")

#aerosol slow response check
vardiff,varttest = getstats(var2mamts-var3mamts,var4mamts-var5mamts)

plt.clf()
fig = plt.figure(3)
ax = fig.add_subplot(111)

map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
map.drawcoastlines()
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1],20)
meridians = np.arange(lonbounds[0],lonbounds[1],20)
map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6)
map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6)
#x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
x, y = map(mlons, mlats)
#print(mlons)
#print(lons[lonli:lonui+1])
#print(x)
clevs = np.arange(-4.75,5.25,0.5)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", "..."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response MAM "+varstr+" changes check",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_mam_prect_SEA_contour_slowaerosol_check.pdf")

