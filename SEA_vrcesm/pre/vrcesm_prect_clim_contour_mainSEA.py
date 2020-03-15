#this script is used to compare vrcesm against observations
#here mean climatology is presented
#by Harry Li

#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
import netCDF4
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import datetime as datetime
import pandas as pd

#set up pre observations directory and filename
obsdir1 = "/scratch/d/dylan/harryli/obsdataset/CRU/"
obsdir2 = "/scratch/d/dylan/harryli/obsdataset/GPCP/"
obsdir3 = "/scratch/d/dylan/harryli/obsdataset/GPCC/"
obsdir4 = "/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/monthly/"
obsdir5 = "/scratch/d/dylan/harryli/obsdataset/APHRODITE/MA/"
refdir  = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/dst/"

obsfname1 = "cru_ts3.21.1901.2012.pre_orig_mmday.nc"
obsfname2 = "gpcp_cdr_v23rB1_197901-201608.nc"
obsfname3 = "precip.mon.total.v7.nc"
obsfname4 = "era_interim_pre_monthly_197901-200512.nc"
obsfname5 = "APHRO_MA_025deg_V1101.1951-2007.nc"

obs1_in = Dataset(obsdir1+obsfname1)
obs2_in = Dataset(obsdir2+obsfname2)
obs3_in = Dataset(obsdir3+obsfname3)
obs4_in = Dataset(obsdir4+obsfname4)
obs5_in = Dataset(obsdir5+obsfname5)

#set up cesm data directories and filenames
dir1 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
dir2 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
dir3 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
dir4 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f19_f19_AMIP_1979_to_2005/atm/hist/"
#dir4 = "/scratch/d/dylan/harryli/gpcdata/cesm1.2/gpc_cesm1_2_2/archive/AMIP1979control_2deg/atm/hist/"

fname1 = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname2 = "fv09_PREC_ne30_ne30_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname3 = "PREC_f09_f09_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname4 = "PREC_f19_f19_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
#fname4 = "diag_AMIP1979control_2deg.cam.h0.1979_2005.nc"

mod1_in = Dataset(dir1+fname1)
mod2_in = Dataset(dir2+fname2)
mod3_in = Dataset(dir3+fname3)
mod4_in = Dataset(dir4+fname4)

model_list = ['vrseasia','ne30','fv0.9x1.25','fv1.9x2.5']

#set up land fraction
dir_landfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
f1_landfrc = "USGS-gtopo30_0.23x0.31_remap_c061107.nc"
f2_landfrc = "USGS-gtopo30_0.9x1.25_remap_c051027.nc"
f3_landfrc = "USGS-gtopo30_0.9x1.25_remap_c051027.nc"
f4_landfrc = "USGS-gtopo30_1.9x2.5_remap_c050602.nc"

mod1_lndfrc = Dataset(dir_landfrc+f1_landfrc)
mod2_lndfrc = Dataset(dir_landfrc+f2_landfrc)
mod3_lndfrc = Dataset(dir_landfrc+f3_landfrc)
mod4_lndfrc = Dataset(dir_landfrc+f4_landfrc)

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/contour/"
outlog = open(outdir+"vrcesm_prect_clim_line_plot_output.log", "w")

#define inital year and end year
iniyear = 1980
endyear = 2005
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01','%Y-%m-%d')
enddate = datetime.datetime.strptime(str(endyear)+'-12-31','%Y-%m-%d')
yearts  = np.arange(iniyear,endyear+1)
print(yearts)

#define mainland Southeast Asia region
reg_lats = [10,25]
reg_lons = [100,110]

#define the contour plot region
latbounds = [ -15 , 55 ]
lonbounds = [ 60 , 150 ]

#define days of each month in the leap year and normal year
leapmonthday = np.array([31,29,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
monthday = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
month = np.arange(1,13,1)
mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#define model names
models = ['vrseasia','ne30','fv0.9x1.25','fv1.9x2.5']

###########################################################
#read data
###########################################################

#obs  
#CRU
obs1_lats = obs1_in.variables['lat'][:]
obs1_lons = obs1_in.variables['lon'][:]

obs1_lat_1 = np.argmin( np.abs( obs1_lats - latbounds[0] ) )
obs1_lat_2 = np.argmin( np.abs( obs1_lats - latbounds[1] ) )
obs1_lon_1 = np.argmin( np.abs( obs1_lons - lonbounds[0] ) )
obs1_lon_2 = np.argmin( np.abs( obs1_lons - lonbounds[1] ) )

obs1_var = obs1_in.variables['pre'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12, obs1_lat_1 : obs1_lat_2 + 1, obs1_lon_1 : obs1_lon_2 + 1]
obs1_time = netCDF4.num2date(obs1_in.variables['time'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12],obs1_in.variables['time'].units)
obs1_time = pd.to_datetime(obs1_time)
print(obs1_time)

#GPCP
obs2_lats = obs2_in.variables['latitude'][:]
obs2_lons = obs2_in.variables['longitude'][:]

obs2_lat_1 = np.argmin( np.abs( obs2_lats - latbounds[0] ) )
obs2_lat_2 = np.argmin( np.abs( obs2_lats - latbounds[1] ) )
obs2_lon_1 = np.argmin( np.abs( obs2_lons - lonbounds[0] ) )
obs2_lon_2 = np.argmin( np.abs( obs2_lons - lonbounds[1] ) )

obs2_var = obs2_in.variables['precip'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, obs2_lat_1 : obs2_lat_2 + 1, obs2_lon_1 : obs2_lon_2 + 1]
obs2_time = netCDF4.num2date(obs2_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],obs2_in.variables['time'].units)
obs2_time = pd.to_datetime(obs2_time)

#GPCC
obs3_lats = obs3_in.variables['lat'][:]
obs3_lons = obs3_in.variables['lon'][:]

obs3_lat_1 = np.argmin( np.abs( obs3_lats - latbounds[0] ) )
obs3_lat_2 = np.argmin( np.abs( obs3_lats - latbounds[1] ) )
obs3_lon_1 = np.argmin( np.abs( obs3_lons - lonbounds[0] ) )
obs3_lon_2 = np.argmin( np.abs( obs3_lons - lonbounds[1] ) )

obs3_var = obs3_in.variables['precip'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12, obs3_lat_2 : obs3_lat_1 + 1, obs3_lon_1 : obs3_lon_2 + 1]
obs3_time = netCDF4.num2date(obs3_in.variables['time'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12],obs3_in.variables['time'].units)
obs3_time = pd.to_datetime(obs3_time)

obs3_lats = obs3_lats[::-1]
obs3_var = obs3_var[:,::-1,:]

for k in range(12):
    obs3_var[k::12,:,:] = obs3_var[k::12,:,:] / monthday[k]


#ERA_interim
time_var = obs4_in.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)
dtime = pd.to_datetime(dtime)
select_dtime = (dtime>=inidate)&(dtime<=enddate)&(~((dtime.month==2)&(dtime.day==29)))

obs4_lats = obs4_in.variables['latitude'][:]
obs4_lons = obs4_in.variables['longitude'][:]

obs4_lat_1 = np.argmin( np.abs( obs4_lats - latbounds[0] ) )
obs4_lat_2 = np.argmin( np.abs( obs4_lats - latbounds[1] ) )
obs4_lon_1 = np.argmin( np.abs( obs4_lons - lonbounds[0] ) )
obs4_lon_2 = np.argmin( np.abs( obs4_lons - lonbounds[1] ) )

obs4_var = obs4_in.variables['tp'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, obs4_lat_2 : obs4_lat_1 + 1, obs4_lon_1 : obs4_lon_2 + 1]
obs4_time = netCDF4.num2date(obs4_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],obs4_in.variables['time'].units)
obs4_time = pd.to_datetime(obs4_time)

#obs4_var = obs4_in.variables['tp'][select_dtime, obs4_lat_2 : obs4_lat_1 + 1, obs4_lon_1 : obs4_lon_2 + 1]
obs4_lats = obs4_lats[::-1]
obs4_var = obs4_var[:,::-1,:] * 1000    #convert from m to mm


#APHRODITE
obs5_lats = obs5_in.variables['latitude'][:]
obs5_lons = obs5_in.variables['longitude'][:]

obs5_lat_1 = np.argmin( np.abs( obs5_lats - latbounds[0] ) )
obs5_lat_2 = np.argmin( np.abs( obs5_lats - latbounds[1] ) )
obs5_lon_1 = np.argmin( np.abs( obs5_lons - lonbounds[0] ) )
obs5_lon_2 = np.argmin( np.abs( obs5_lons - lonbounds[1] ) )

obs5_time = netCDF4.num2date(obs5_in.variables['time'][:],obs5_in.variables['time'].units)
obs5_time = pd.to_datetime(obs5_time)
select_dtime = (obs5_time>=inidate)&(obs5_time<=enddate)&(~((obs5_time.month==2)&(obs5_time.day==29)))
obs5_var = obs5_in.variables['precip'][select_dtime, obs5_lat_1 : obs5_lat_2 + 1, obs5_lon_1 : obs5_lon_2 + 1]
obs5_time = obs5_time[select_dtime]

obs5_var = np.array(obs5_var)
obs5_var = np.ma.masked_where(obs5_var<0., obs5_var)
print(obs5_var.shape)
print(obs5_time.shape)


#model  
#vrseasia
mod1_lats = mod1_in.variables['lat'][:]
mod1_lons = mod1_in.variables['lon'][:]

mod1_lat_1 = np.argmin( np.abs( mod1_lats - latbounds[0] ) )
mod1_lat_2 = np.argmin( np.abs( mod1_lats - latbounds[1] ) )
mod1_lon_1 = np.argmin( np.abs( mod1_lons - lonbounds[0] ) )
mod1_lon_2 = np.argmin( np.abs( mod1_lons - lonbounds[1] ) )

mod1_var = mod1_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]
mod1_time = netCDF4.num2date(mod1_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],mod1_in.variables['time'].units)
mod1_time = pd.to_datetime(mod1_time)
mod1_var = mod1_var * 86400 *1000
mod1_frc = mod1_lndfrc.variables['LANDFRAC'][mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]


#ne30
mod2_lats = mod2_in.variables['lat'][:]
mod2_lons = mod2_in.variables['lon'][:]

mod2_lat_1 = np.argmin( np.abs( mod2_lats - latbounds[0] ) )
mod2_lat_2 = np.argmin( np.abs( mod2_lats - latbounds[1] ) )
mod2_lon_1 = np.argmin( np.abs( mod2_lons - lonbounds[0] ) )
mod2_lon_2 = np.argmin( np.abs( mod2_lons - lonbounds[1] ) )

mod2_var = mod2_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]
mod2_time = netCDF4.num2date(mod2_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],mod2_in.variables['time'].units)
mod2_time = pd.to_datetime(mod2_time)
mod2_var = mod2_var * 86400 *1000
mod2_frc = mod2_lndfrc.variables['LANDFRAC'][mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]

#fv0.9x1.25
mod3_lats = mod3_in.variables['lat'][:]
mod3_lons = mod3_in.variables['lon'][:]

mod3_lat_1 = np.argmin( np.abs( mod3_lats - latbounds[0] ) )
mod3_lat_2 = np.argmin( np.abs( mod3_lats - latbounds[1] ) )
mod3_lon_1 = np.argmin( np.abs( mod3_lons - lonbounds[0] ) )
mod3_lon_2 = np.argmin( np.abs( mod3_lons - lonbounds[1] ) )

mod3_var = mod3_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod3_lat_1 : mod3_lat_2 + 1, mod3_lon_1 : mod3_lon_2 + 1]
mod3_time = netCDF4.num2date(mod3_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],mod3_in.variables['time'].units)
mod3_time = pd.to_datetime(mod3_time)
mod3_var = mod3_var * 86400 *1000
mod3_frc = mod3_lndfrc.variables['LANDFRAC'][mod3_lat_1 : mod3_lat_2 + 1, mod3_lon_1 : mod3_lon_2 + 1]

#fv1.9x2.5
mod4_lats = mod4_in.variables['lat'][:]
mod4_lons = mod4_in.variables['lon'][:]

mod4_lat_1 = np.argmin( np.abs( mod4_lats - latbounds[0] ) )
mod4_lat_2 = np.argmin( np.abs( mod4_lats - latbounds[1] ) )
mod4_lon_1 = np.argmin( np.abs( mod4_lons - lonbounds[0] ) )
mod4_lon_2 = np.argmin( np.abs( mod4_lons - lonbounds[1] ) )

#print mod4_lats
#print mod4_lat_1, mod4_lat_2

mod4_var = mod4_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
mod4_time = netCDF4.num2date(mod4_in.variables['time'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12],mod4_in.variables['time'].units)
mod4_time = pd.to_datetime(mod4_time)
mod4_var = mod4_var * 86400 *1000
mod4_frc = mod4_lndfrc.variables['LANDFRAC'][mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
#print(mod4_frc[:,:])
print("-------------------")


###########################################################
#calculate statistics and regrid
###########################################################

print("Calculate the statistics and plot...")

#plot against CRU

print("Plot against CRU...")
 
ref_lats = obs1_lats[obs1_lat_1 : obs1_lat_2 + 1]
ref_lons = obs1_lons[obs1_lon_1 : obs1_lon_2 + 1]


for imon in range(12):
    imon = imon + 1
    ref_time = obs1_time[(obs1_time.month==imon)]
    ref_var  = obs1_var[(obs1_time.month==imon),:,:]
    #print(ref_time)
    print(ref_var.shape)
    
    ref_var = np.mean(ref_var,axis=0)

    #print(mod1_var[(mod1_time.month==imon),:,:].shape)
    mod1_var_mean = np.mean(mod1_var[(mod1_time.month==imon),:,:],axis=0) 
    mod2_var_mean = np.mean(mod2_var[(mod2_time.month==imon),:,:],axis=0)
    mod3_var_mean = np.mean(mod3_var[(mod3_time.month==imon),:,:],axis=0)
    mod4_var_mean = np.mean(mod4_var[(mod4_time.month==imon),:,:],axis=0)

    lons_sub, lats_sub = np.meshgrid(ref_lons[:], ref_lats[:])
    plt1_var_diff = basemap.interp(mod1_var_mean, mod1_lons[mod1_lon_1 : mod1_lon_2 + 1], mod1_lats[mod1_lat_1 : mod1_lat_2 + 1], lons_sub, lats_sub, order=1) - ref_var
    plt1_lats = ref_lats
    plt1_lons = ref_lons

    lons_sub, lats_sub = np.meshgrid(mod2_lons[mod2_lon_1 : mod2_lon_2 + 1], mod2_lats[mod2_lat_1 : mod2_lat_2 + 1])
    plt2_var_diff = mod2_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt2_lats = mod2_lats[mod2_lat_1 : mod2_lat_2 + 1]
    plt2_lons = mod2_lons[mod2_lon_1 : mod2_lon_2 + 1]

    lons_sub, lats_sub = np.meshgrid(mod3_lons[mod3_lon_1 : mod3_lon_2 + 1], mod3_lats[mod3_lat_1 : mod3_lat_2 + 1])
    plt3_var_diff = mod3_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt3_lats = mod3_lats[mod3_lat_1 : mod3_lat_2 + 1]
    plt3_lons = mod3_lons[mod3_lon_1 : mod3_lon_2 + 1]

    lons_sub, lats_sub = np.meshgrid(mod4_lons[mod4_lon_1 : mod4_lon_2 + 1], mod4_lats[mod4_lat_1 : mod4_lat_2 + 1])
    plt4_var_diff = mod4_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt4_lats = mod4_lats[mod4_lat_1 : mod4_lat_2 + 1]
    plt4_lons = mod4_lons[mod4_lon_1 : mod4_lon_2 + 1]
    print(ref_lons)
    print(mod4_lons[mod4_lon_1 : mod4_lon_2 + 1])
    print(ref_lats)
    print(mod4_lats[mod4_lat_1 : mod4_lat_2 + 1])
    print(basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1))
    
    #plot figure    
    plt.clf()
    fig = plt.figure()

    #mod1 diff
    ax1 = fig.add_subplot(221)
    ax1.set_title(models[0],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt1_lons, plt1_lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-3.,3.1,0.3)
    cs = map.contourf(x,y,plt1_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    #mod2 diff
    ax2 = fig.add_subplot(222)
    ax2.set_title(models[1],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt2_lons, plt2_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt2_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax2.xaxis.set_tick_params(labelsize=5)
    ax2.yaxis.set_tick_params(labelsize=5)

    #mod3 diff
    ax3 = fig.add_subplot(223)
    ax3.set_title(models[2],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt3_lons, plt3_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt3_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.yaxis.set_tick_params(labelsize=5)

    #mod4 diff
    ax4 = fig.add_subplot(224)
    ax4.set_title(models[3],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt4_lons, plt4_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt4_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax4.xaxis.set_tick_params(labelsize=5)
    ax4.yaxis.set_tick_params(labelsize=5)

    # add colorbar.
    fig.subplots_adjust(bottom=0.2,wspace = 0.2,hspace = 0.2)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs,cax = cbar_ax,orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('mm/day',fontsize=5)

    # add title
    fig.suptitle(mname[imon-1]+" mean prect difference",fontsize=10,y=0.95)
    plt.savefig(outdir+"vrseasia_prect_diff_SEA_contour_vsCRU_"+str(imon)+".pdf",bbox_inches='tight')
    plt.close(fig)
    





#plot against APHRODITE

print("Plot against APHRODITE...")

ref_lats = obs5_lats[obs5_lat_1 : obs5_lat_2 + 1]
ref_lons = obs5_lons[obs5_lon_1 : obs5_lon_2 + 1]


for imon in range(12):
    imon = imon + 1
    ref_time = obs5_time[(obs5_time.month==imon)]
    ref_var  = obs5_var[(obs5_time.month==imon),:,:]
    #print(ref_time)
    print(ref_var.shape)

    ref_var = np.mean(ref_var,axis=0)

    #print(mod1_var[(mod1_time.month==imon),:,:].shape)
    mod1_var_mean = np.mean(mod1_var[(mod1_time.month==imon),:,:],axis=0)
    mod2_var_mean = np.mean(mod2_var[(mod2_time.month==imon),:,:],axis=0)
    mod3_var_mean = np.mean(mod3_var[(mod3_time.month==imon),:,:],axis=0)
    mod4_var_mean = np.mean(mod4_var[(mod4_time.month==imon),:,:],axis=0)

    lons_sub, lats_sub = np.meshgrid(ref_lons[:], ref_lats[:])
    plt1_var_diff = basemap.interp(mod1_var_mean, mod1_lons[mod1_lon_1 : mod1_lon_2 + 1], mod1_lats[mod1_lat_1 : mod1_lat_2 + 1], lons_sub, lats_sub, order=1) - ref_var
    plt1_lats = ref_lats
    plt1_lons = ref_lons

    lons_sub, lats_sub = np.meshgrid(mod2_lons[mod2_lon_1 : mod2_lon_2 + 1], mod2_lats[mod2_lat_1 : mod2_lat_2 + 1])
    plt2_var_diff = mod2_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt2_lats = mod2_lats[mod2_lat_1 : mod2_lat_2 + 1]
    plt2_lons = mod2_lons[mod2_lon_1 : mod2_lon_2 + 1]

    lons_sub, lats_sub = np.meshgrid(mod3_lons[mod3_lon_1 : mod3_lon_2 + 1], mod3_lats[mod3_lat_1 : mod3_lat_2 + 1])
    plt3_var_diff = mod3_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt3_lats = mod3_lats[mod3_lat_1 : mod3_lat_2 + 1]
    plt3_lons = mod3_lons[mod3_lon_1 : mod3_lon_2 + 1]

    lons_sub, lats_sub = np.meshgrid(mod4_lons[mod4_lon_1 : mod4_lon_2 + 1], mod4_lats[mod4_lat_1 : mod4_lat_2 + 1])
    plt4_var_diff = mod4_var_mean - basemap.interp(ref_var, ref_lons, ref_lats, lons_sub, lats_sub, order=1)
    plt4_lats = mod4_lats[mod4_lat_1 : mod4_lat_2 + 1]
    plt4_lons = mod4_lons[mod4_lon_1 : mod4_lon_2 + 1]


    #plot figure    
    plt.clf()
    fig = plt.figure()

    #mod1 diff
    ax1 = fig.add_subplot(221)
    ax1.set_title(models[0],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt1_lons, plt1_lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-3.,3.1,0.3)
    cs = map.contourf(x,y,plt1_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    #mod2 diff
    ax2 = fig.add_subplot(222)
    ax2.set_title(models[1],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt2_lons, plt2_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt2_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax2.xaxis.set_tick_params(labelsize=5)
    ax2.yaxis.set_tick_params(labelsize=5)

    #mod3 diff
    ax3 = fig.add_subplot(223)
    ax3.set_title(models[2],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt3_lons, plt3_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt3_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.yaxis.set_tick_params(labelsize=5)

    #mod4 diff
    ax4 = fig.add_subplot(224)
    ax4.set_title(models[3],fontsize=6,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(plt4_lons, plt4_lats)
    x, y = map(mlons, mlats)
    cs = map.contourf(x,y,plt4_var_diff,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")
    ax4.xaxis.set_tick_params(labelsize=5)
    ax4.yaxis.set_tick_params(labelsize=5)

    # add colorbar.
    fig.subplots_adjust(bottom=0.2,wspace = 0.2,hspace = 0.2)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs,cax = cbar_ax,orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('mm/day',fontsize=5)

    # add title
    fig.suptitle(mname[imon-1]+" mean prect difference",fontsize=10,y=0.95)
    plt.savefig(outdir+"vrseasia_prect_diff_SEA_contour_vsAPHRODITE_"+str(imon)+".pdf",bbox_inches='tight')
    plt.close(fig)



