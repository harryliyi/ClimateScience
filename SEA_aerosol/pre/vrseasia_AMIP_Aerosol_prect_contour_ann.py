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
iniyear = 2
endyear = 50

#define the contour plot region
latbounds = [ -20 , 50 ]
lonbounds = [ 40 , 160 ]

#define Southeast region
reg_lats = [10, 20]
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

    return vardiff,np.abs(varttest)

################################################################################################
#S1-Plot climatological data
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

# latitude/longitude  lower and upper regional index
reg_latli = np.abs(lats - reg_lats[0]).argmin()
reg_latui = np.abs(lats - reg_lats[1]).argmin()

reg_lonli = np.abs(lons - reg_lons[0]).argmin()
reg_lonui = np.abs(lons - reg_lons[1]).argmin()

#read the monthly data
var1 = fdata1.variables[varname][ (iniyear-1)*12 : (endyear)*12 , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
var2 = fdata2.variables[varname][ (iniyear-1)*12 : (endyear)*12 , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
var3 = fdata3.variables[varname][ (iniyear-1)*12 : (endyear)*12 , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
var4 = fdata4.variables[varname][ (iniyear-1)*12 : (endyear)*12 , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
var5 = fdata5.variables[varname][ (iniyear-1)*12 : (endyear)*12 , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
#print(var1.shape)

var1ts = np.mean(np.mean(var1,axis = 2),axis = 1)*86400*1000
var2ts = np.mean(np.mean(var2,axis = 2),axis = 1)*86400*1000
var3ts = np.mean(np.mean(var3,axis = 2),axis = 1)*86400*1000
var4ts = np.mean(np.mean(var4,axis = 2),axis = 1)*86400*1000
var5ts = np.mean(np.mean(var5,axis = 2),axis = 1)*86400*1000

#plot timeseries
var1season = np.array([],dtype=float)
var2season = np.array([],dtype=float)
var3season = np.array([],dtype=float)
var4season = np.array([],dtype=float)
var5season = np.array([],dtype=float)

var1tsstd = np.array([],dtype=float)
var2tsstd = np.array([],dtype=float)
var3tsstd = np.array([],dtype=float)
var4tsstd = np.array([],dtype=float)
var5tsstd = np.array([],dtype=float)

for k in np.arange(0,12,1):
    var1season = np.append(var1season,np.mean(var1ts[k::12]))
    var2season = np.append(var2season,np.mean(var2ts[k::12]))
    var3season = np.append(var3season,np.mean(var3ts[k::12]))
    var4season = np.append(var4season,np.mean(var4ts[k::12]))
    var5season = np.append(var5season,np.mean(var5ts[k::12]))

    var1tsstd = np.append(var1tsstd,np.std(var1ts[k::12]))
    var2tsstd = np.append(var2tsstd,np.std(var2ts[k::12]))
    var3tsstd = np.append(var3tsstd,np.std(var3ts[k::12]))
    var4tsstd = np.append(var4tsstd,np.std(var4ts[k::12]))
    var5tsstd = np.append(var5tsstd,np.std(var5ts[k::12]))

#plot climatology for each case
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)

plt.plot(month,var1season, color='black', marker='o', markersize=2, linestyle='solid',linewidth=2, label = case1)
plt.errorbar(month, var1season, yerr = var1tsstd, fmt='o',markersize=2,elinewidth=1, color = 'black')

plt.plot(month,var2season, color='red', marker='o', markersize=2, linestyle='solid',linewidth=2, label = case2)
plt.errorbar(month, var2season, yerr = var2tsstd, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.plot(month,var3season, color='green', marker='o', markersize=2, linestyle='solid',linewidth=2, label = case3)
plt.errorbar(month, var3season, yerr = var3tsstd, fmt='o',markersize=2,elinewidth=1, color = 'green')

plt.plot(month,var4season, color='blue', marker='o', markersize=2, linestyle='solid',linewidth=2, label = case4)
plt.errorbar(month, var4season, yerr = var4tsstd, fmt='o',markersize=2,elinewidth=1, color = 'blue')

plt.plot(month,var5season, color='brown', marker='o', markersize=2, linestyle='solid',linewidth=2, label = case5)
plt.errorbar(month, var5season, yerr = var5tsstd, fmt='o',markersize=2,elinewidth=1, color = 'brown')

plt.legend(loc='upper left',prop={'size': 6})
plt.xticks(month,mname)
plt.title("Southeast Asia "+varstr+" climatology",fontsize=12)
plt.ylabel(varstr+" (mm/day)")
plt.xlabel("Month")
plt.savefig(outdir+"vrseasia_aerosol_amip_prect_mainlandSEA_monclim_allcases.pdf")


#plot climatology for each forcing
plt.clf()
fig = plt.figure(2)
ax = fig.add_subplot(111)

n1 = endyear-iniyear+1
n2 = endyear-iniyear+1

vardiff = var2season-var1season
varstde = np.sqrt(var2tsstd**2/n1+var1tsstd**2/n2) 
plt.plot(month,vardiff, color='black', linestyle='solid',linewidth=2, label = 'All forcing')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'black')

vardiff = var5season-var1season
varstde = np.sqrt(var5tsstd**2/n1+var1tsstd**2/n2)
plt.plot(month,vardiff, color='blue', linestyle='solid',linewidth=2, label = 'GHG and natral forcing')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'blue')

vardiff = var2season-var4season
varstde = np.sqrt(var2tsstd**2/n1+var4tsstd**2/n2)
plt.plot(month,vardiff, color='green', linestyle='dashed',linewidth=2, label = 'Aerosol fast response at SST2000s')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'green')

vardiff = var3season-var5season
varstde = np.sqrt(var3tsstd**2/n1+var5tsstd**2/n2)
plt.plot(month,vardiff, color='cyan', linestyle='dashed',linewidth=2, label = 'Aerosol fast response at SST2000s without aerosols')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'cyan')


vardiff = var4season-var5season
varstde = np.sqrt(var4tsstd**2/n1+var5tsstd**2/n2)
plt.plot(month,vardiff, color='magenta', linestyle='dotted',linewidth=2, label = 'Aerosol slow response under AER1950s')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'magenta')

vardiff = var2season-var3season
varstde = np.sqrt(var2tsstd**2/n1+var3tsstd**2/n2)
plt.plot(month,var2season-var3season, color='red', linestyle='dotted',linewidth=2, label = 'Aerosol slow response under AER2000s')
plt.errorbar(month, vardiff, yerr = varstde, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.legend(loc='upper left',prop={'size': 6})
plt.xticks(month,mname)
plt.title("Southeast Asia "+varstr+" climatology",fontsize=12)
plt.ylabel(varstr+" (mm/day)")
plt.xlabel("Month")
plt.savefig(outdir+"vrseasia_aerosol_amip_prect_mainlandSEA_monclim_allforcings.pdf")


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
#annual means
var1ann = np.mean(var1,axis=0)
var2ann = np.mean(var2,axis=0)
var3ann = np.mean(var3,axis=0)
var4ann = np.mean(var4,axis=0)
var5ann = np.mean(var5,axis=0)

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
cs = map.contourf(x,y,var1ann,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case1+" Annual "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_case1.pdf")

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
cs = map.contourf(x,y,var2ann,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case2+" Annual "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_case2.pdf")

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
cs = map.contourf(x,y,var3ann,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case3+" Annual "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_case3.pdf")

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
cs = map.contourf(x,y,var4ann,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case4+" Annual "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_case4.pdf")

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
cs = map.contourf(x,y,var5ann,cmap=cm.YlGnBu,alpha = 0.9)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(case5+" Annual "+varstr,fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_case5.pdf")


#################################################################################################
#annual mean changes
var1annts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var2annts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var3annts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var4annts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
var5annts = np.zeros(((endyear-iniyear+1),latui-latli+1,lonui-lonli+1))
print(var1annts.shape)

for iyear in range(endyear-iniyear+1):
    var1annts[iyear,:,:] = np.mean(var1[iyear*12:(iyear+1)*12,:,:],axis=0)
    var2annts[iyear,:,:] = np.mean(var2[iyear*12:(iyear+1)*12,:,:],axis=0)
    var3annts[iyear,:,:] = np.mean(var3[iyear*12:(iyear+1)*12,:,:],axis=0)
    var4annts[iyear,:,:] = np.mean(var4[iyear*12:(iyear+1)*12,:,:],axis=0)
    var5annts[iyear,:,:] = np.mean(var5[iyear*12:(iyear+1)*12,:,:],axis=0)

#all forcings
vardiff,varttest = getstats(var2annts,var1annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("All forcings Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_allforcings.pdf")

#GHG and natural forcing
vardiff,varttest = getstats(var5annts,var1annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("GHG and natural forcings Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_GHGforcings.pdf")

#all aerosol forcing
vardiff,varttest = getstats(var2annts,var5annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("All Aerosol forcings Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_allaerosols.pdf")

#aerosol fast response1
vardiff,varttest = getstats(var2annts,var4annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_fastaerosol1.pdf")

#aerosol fast response2
vardiff,varttest = getstats(var3annts,var5annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_fastaerosol2.pdf")

#aerosol fast response check
vardiff,varttest = getstats(var2annts-var4annts,var3annts-var5annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol fast response Annual "+varstr+" changes check",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_fastaerosol_check.pdf")

#aerosol slow response1
vardiff,varttest = getstats(var2annts,var3annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_slowaerosol1.pdf")

#aerosol slow response2
vardiff,varttest = getstats(var4annts,var5annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response Annual "+varstr+" changes",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_slowaerosol2.pdf")

#aerosol slow response check
vardiff,varttest = getstats(var2annts-var3annts,var4annts-var5annts)

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
clevs = np.arange(-1.4,1.5,0.2)
cs = map.contourf(x,y,vardiff,clevs,cmap=cm.BrBG,alpha = 0.9)
levels = [0., 2.01, varttest.max()]
csm = plt.contourf(x,y,varttest,levels=levels,colors='none',hatches=["", ".."], alpha=0)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.ax.tick_params(labelsize=5)
cbar.set_label('mm')
# add title
plt.title("Aerosol slow response Annual "+varstr+" changes check",fontsize=12)
plt.savefig(outdir+"vrseasia_aerosol_amip_annual_prect_SEA_contour_slowaerosol_check.pdf")

