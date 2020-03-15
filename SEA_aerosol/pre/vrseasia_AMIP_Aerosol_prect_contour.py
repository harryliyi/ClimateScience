#This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

#S1-read climatological data from the Exp
#S2-calculating the aerosol forcing and statistics
#S3-plot the data 
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
latbounds = [ -15 , 40 ]
lonbounds = [ 60 , 130 ]

#define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

#month series
month = np.arange(1,13,1)
mname = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

################################################################################################
#S1-Read the data
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

# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

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
print(var1.shape)

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

plt.legend(loc='upper left')
plt.xticks(month,mname)
plt.title("Southeast Asia "+varstr+" climatology")
plt.ylabel(varstr+" (mm/day)")
plt.xlabel("Month")
plt.savefig(outdir+"vrseasia_aerosol_amip_prect_mainlandSEA_monclim_allcases.pdf")


#plot climatology for each forcing
plt.clf()
fig = plt.figure(2)
ax = fig.add_subplot(111)

plt.plot(month,var2season-var1season, color='black', linestyle='solid',linewidth=2, label = 'All forcing')
plt.plot(month,var5season-var1season, color='blue', linestyle='solid',linewidth=2, label = 'GHG and natral forcing')
plt.plot(month,var2season-var4season, color='green', linestyle='dashed',linewidth=2, label = 'Aerosol fast response at SST2000s')
plt.plot(month,var3season-var5season, color='cyan', linestyle='dashed',linewidth=2, label = 'Aerosol fast response at SST2000s without aerosols')
plt.plot(month,var4season-var5season, color='magenta', linestyle='dotted',linewidth=2, label = 'Aerosol slow response under AER1950s')
plt.plot(month,var2season-var3season, color='red', linestyle='dotted',linewidth=2, label = 'Aerosol slow response under AER2000s')

plt.legend(loc='upper left')
plt.xticks(month,mname)
plt.title("Southeast Asia "+varstr+" climatology")
plt.ylabel(varstr+" (mm/day)")
plt.xlabel("Month")
plt.savefig(outdir+"vrseasia_aerosol_amip_prect_mainlandSEA_monclim_allforcings.pdf")


