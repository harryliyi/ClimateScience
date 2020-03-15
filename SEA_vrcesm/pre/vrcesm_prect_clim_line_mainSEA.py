#this script is used to compare vrcesm against observations
#here mean climatology is presented
#by Harry Li


#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
import netCDF4
from mpl_toolkits.basemap import Basemap
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
#obsdir4 = "/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/daily/"
obsdir5 = "/scratch/d/dylan/harryli/obsdataset/APHRODITE/MA/"
refdir  = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/dst/"

obsfname1 = "cru_ts3.21.1901.2012.pre_orig_mmday.nc"
obsfname2 = "gpcp_cdr_v23rB1_197901-201608.nc"
obsfname3 = "precip.mon.total.v7.nc"
obsfname4 = "era_interim_pre_monthly_197901-200512.nc"
#obsfname4 = "era_interim_tp_daily_19790101-20051231.nc"
obsfname5 = "APHRO_MA_025deg_V1101.1980.nc"

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
fname3 = "fv09_PREC_f09_f09_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname4 = "fv19_PREC_f19_f19_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/"
outlog = open(outdir+"vrcesm_prect_clim_line_plot_output.log", "w")

#define inital year and end year
iniyear = 1980
endyear = 2005
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01 12:00','%Y-%m-%d %H:%M')
enddate = datetime.datetime.strptime(str(endyear)+'-12-31 12:00','%Y-%m-%d %H:%M')
yearts  = np.arange(iniyear,endyear+1)
print(yearts)

#define mainland Southeast Asia region
reg_lats = [10,25]
reg_lons = [100,110]

#define days of each month in the leap year and normal year
leapmonthday = np.array([31,29,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
monthday = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
month = np.arange(1,13,1)
mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

###########################################################
#read data
###########################################################
#obs  
#CRU
obs1_lats = obs1_in.variables['lat'][:]
obs1_lons = obs1_in.variables['lon'][:]

obs1_lat_1 = np.argmin( np.abs( obs1_lats - reg_lats[0] ) )
obs1_lat_2 = np.argmin( np.abs( obs1_lats - reg_lats[1] ) )
obs1_lon_1 = np.argmin( np.abs( obs1_lons - reg_lons[0] ) )
obs1_lon_2 = np.argmin( np.abs( obs1_lons - reg_lons[1] ) )

obs1_var = obs1_in.variables['pre'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12, obs1_lat_1 : obs1_lat_2 + 1, obs1_lon_1 : obs1_lon_2 + 1]

#GPCP
obs2_lats = obs2_in.variables['latitude'][:]
obs2_lons = obs2_in.variables['longitude'][:]

obs2_lat_1 = np.argmin( np.abs( obs2_lats - reg_lats[0] ) )
obs2_lat_2 = np.argmin( np.abs( obs2_lats - reg_lats[1] ) )
obs2_lon_1 = np.argmin( np.abs( obs2_lons - reg_lons[0] ) )
obs2_lon_2 = np.argmin( np.abs( obs2_lons - reg_lons[1] ) )

obs2_var = obs2_in.variables['precip'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, obs2_lat_1 : obs2_lat_2 + 1, obs2_lon_1 : obs2_lon_2 + 1]

#GPCC
obs3_lats = obs3_in.variables['lat'][:]
obs3_lons = obs3_in.variables['lon'][:]

obs3_lat_1 = np.argmin( np.abs( obs3_lats - reg_lats[0] ) )
obs3_lat_2 = np.argmin( np.abs( obs3_lats - reg_lats[1] ) )
obs3_lon_1 = np.argmin( np.abs( obs3_lons - reg_lons[0] ) )
obs3_lon_2 = np.argmin( np.abs( obs3_lons - reg_lons[1] ) )

obs3_var = obs3_in.variables['precip'][(iniyear-1901)*12 : (endyear-1901 + 1) * 12, obs3_lat_2 : obs3_lat_1 + 1, obs3_lon_1 : obs3_lon_2 + 1]
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

obs4_lat_1 = np.argmin( np.abs( obs4_lats - reg_lats[0] ) )
obs4_lat_2 = np.argmin( np.abs( obs4_lats - reg_lats[1] ) )
obs4_lon_1 = np.argmin( np.abs( obs4_lons - reg_lons[0] ) )
obs4_lon_2 = np.argmin( np.abs( obs4_lons - reg_lons[1] ) )

obs4_var = obs4_in.variables['tp'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, obs4_lat_2 : obs4_lat_1 + 1, obs4_lon_1 : obs4_lon_2 + 1]
#obs4_var = obs4_in.variables['tp'][select_dtime, obs4_lat_2 : obs4_lat_1 + 1, obs4_lon_1 : obs4_lon_2 + 1]
obs4_lats = obs4_lats[::-1]
obs4_var = obs4_var[:,::-1,:] * 1000    #convert from m to mm
#print(obs4_var[0:24,:,:])
#print(obs3_lat_1)
#print(obs3_lat_2)
#print(obs3_lon_1)
#print(obs3_lon_2)


#APHRODITE
obs5_lats = obs5_in.variables['latitude'][:]
obs5_lons = obs5_in.variables['longitude'][:]

obs5_lat_1 = np.argmin( np.abs( obs5_lats - reg_lats[0] ) )
obs5_lat_2 = np.argmin( np.abs( obs5_lats - reg_lats[1] ) )
obs5_lon_1 = np.argmin( np.abs( obs5_lons - reg_lons[0] ) )
obs5_lon_2 = np.argmin( np.abs( obs5_lons - reg_lons[1] ) )

for iyear in range(iniyear,endyear+1):
    obsfname5 = "APHRO_MA_025deg_V1101."+str(iyear)+".nc"
    obs5_in = Dataset(obsdir5+obsfname5)

    time_var = obs5_in.variables['time']
    #print(time_var[:])
    dtime = netCDF4.num2date(time_var[:],time_var.units)
    dtime = pd.to_datetime(dtime)
    select_dtime = (~((dtime.month==2)&(dtime.day==29)))
    #print(dtime)
    temp = obs5_in.variables['precip'][select_dtime, obs5_lat_1 : obs5_lat_2 + 1, obs5_lon_1 : obs5_lon_2 + 1]
    temp = np.array(temp)
    print(temp.shape)    

    if iyear==iniyear:
        obs5_var = temp
        obs5_time= dtime[select_dtime]
    else:
        obs5_var  = np.concatenate((obs5_var,temp),axis=0)
        obs5_time = np.concatenate((obs5_time,dtime[select_dtime]),axis=0)

obs5_var = np.array(obs5_var)
obs5_var = np.ma.masked_where(obs5_var<0., obs5_var)
#print(obs5_var.shape)
#print(obs5_time.shape)

#calculate the time series
#print(obs5_var[0,:,:])
#print(obs1_var[0,:,:])
obs1_var_ts = np.ma.mean(np.ma.mean(obs1_var,axis=1),axis=1)
obs2_var_ts = np.ma.mean(np.ma.mean(obs2_var,axis=1),axis=1)
obs3_var_ts = np.ma.mean(np.ma.mean(obs3_var,axis=1),axis=1)
obs4_var_ts = np.ma.mean(np.ma.mean(obs4_var,axis=1),axis=1)
obs5_var_ts = np.ma.mean(np.ma.mean(obs5_var,axis=1),axis=1)
print(obs1_var_ts)
#print(obs5_var_ts)

#ERA_interim daily
#obs4_var_ts_pd = pd.DataFrame({
#        'date': dtime[select_dtime],
#        'pre': obs4_var_ts
#    })
#obs4_var_ts_pd = obs4_var_ts_pd.set_index('date')
#print(obs4_var_ts_pd)
#obs4_var_ts_pd = obs4_var_ts_pd.resample('M').mean()
#obs4_var_ts = obs4_var_ts_pd['pre']

#print(obs4_var_ts)

#APHRODITE daily
obs5_var_ts_pd = pd.DataFrame({
        'date': obs5_time,
        'pre': obs5_var_ts
    })
obs5_var_ts_pd = obs5_var_ts_pd.set_index('date')
#print(obs5_var_ts_pd)
obs5_var_ts_pd = obs5_var_ts_pd.resample('M').mean()
obs5_var_ts = obs5_var_ts_pd['pre']
#print(obs5_var_ts)


#model  
#vrseasia
mod1_lats = mod1_in.variables['lat'][:]
mod1_lons = mod1_in.variables['lon'][:]

mod1_lat_1 = np.argmin( np.abs( mod1_lats - reg_lats[0] ) )
mod1_lat_2 = np.argmin( np.abs( mod1_lats - reg_lats[1] ) )
mod1_lon_1 = np.argmin( np.abs( mod1_lons - reg_lons[0] ) )
mod1_lon_2 = np.argmin( np.abs( mod1_lons - reg_lons[1] ) )

mod1_var = mod1_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]
mod1_var = mod1_var * 86400 *1000
mod1_frc = mod1_lndfrc.variables['LANDFRAC'][mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]

#ne30
mod2_lats = mod2_in.variables['lat'][:]
mod2_lons = mod2_in.variables['lon'][:]

mod2_lat_1 = np.argmin( np.abs( mod2_lats - reg_lats[0] ) )
mod2_lat_2 = np.argmin( np.abs( mod2_lats - reg_lats[1] ) )
mod2_lon_1 = np.argmin( np.abs( mod2_lons - reg_lons[0] ) )
mod2_lon_2 = np.argmin( np.abs( mod2_lons - reg_lons[1] ) )

mod2_var = mod2_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]
mod2_var = mod2_var * 86400 *1000
mod2_frc = mod2_lndfrc.variables['LANDFRAC'][mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]

#fv0.9x1.25
mod3_lats = mod3_in.variables['lat'][:]
mod3_lons = mod3_in.variables['lon'][:]

mod3_lat_1 = np.argmin( np.abs( mod3_lats - reg_lats[0] ) )
mod3_lat_2 = np.argmin( np.abs( mod3_lats - reg_lats[1] ) )
mod3_lon_1 = np.argmin( np.abs( mod3_lons - reg_lons[0] ) )
mod3_lon_2 = np.argmin( np.abs( mod3_lons - reg_lons[1] ) )

mod3_var = mod3_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod3_lat_1 : mod3_lat_2 + 1, mod3_lon_1 : mod3_lon_2 + 1]
mod3_var = mod3_var * 86400 *1000
mod3_frc = mod3_lndfrc.variables['LANDFRAC'][mod3_lat_1 : mod3_lat_2 + 1, mod3_lon_1 : mod3_lon_2 + 1]

#fv1.9x2.5
mod4_lats = mod4_in.variables['lat'][:]
mod4_lons = mod4_in.variables['lon'][:]

mod4_lat_1 = np.argmin( np.abs( mod4_lats - reg_lats[0] ) )
mod4_lat_2 = np.argmin( np.abs( mod4_lats - reg_lats[1] ) )
mod4_lon_1 = np.argmin( np.abs( mod4_lons - reg_lons[0] ) )
mod4_lon_2 = np.argmin( np.abs( mod4_lons - reg_lons[1] ) )

#print mod4_lats
#print mod4_lat_1, mod4_lat_2

mod4_var = mod4_in.variables['PRECT'][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
mod4_var = mod4_var * 86400 *1000
mod4_frc = mod4_lndfrc.variables['LANDFRAC'][mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
#print(mod4_frc[:,:])
print("-------------------")

#mask the array to fit CRU data
#print(mod4_var.shape)
mod1_var_mask = mod1_var.copy()
mod2_var_mask = mod2_var.copy()
mod3_var_mask = mod3_var.copy()
mod4_var_mask = mod4_var.copy()

for idx in range((endyear-iniyear+1)*12):
    #if idx==0:
        #print(np.ma.masked_where(mod4_frc<0.5, mod4_var[idx,:,:]))
        #ax = axs[0,0]
    mod1_var_mask[idx,:,:] = np.ma.masked_where(mod1_frc<0.5, mod1_var_mask[idx,:,:])
    mod2_var_mask[idx,:,:] = np.ma.masked_where(mod2_frc<0.5, mod2_var_mask[idx,:,:])
    mod3_var_mask[idx,:,:] = np.ma.masked_where(mod3_frc<0.5, mod3_var_mask[idx,:,:])
    mod4_var_mask[idx,:,:] = np.ma.masked_where(mod4_frc<0.5, mod4_var_mask[idx,:,:])

#print(mod4_var_mask[0,:,:])
print("-------------------")
#print(mod4_var[0,:,:])

#calculate time series
mod1_var_ts = np.sum(np.sum(mod1_var,axis=1),axis=1)/(mod1_lat_2 - mod1_lat_1 + 1)/(mod1_lon_2 - mod1_lon_1 + 1)
mod2_var_ts = np.sum(np.sum(mod2_var,axis=1),axis=1)/(mod2_lat_2 - mod2_lat_1 + 1)/(mod2_lon_2 - mod2_lon_1 + 1)
mod3_var_ts = np.sum(np.sum(mod3_var,axis=1),axis=1)/(mod3_lat_2 - mod3_lat_1 + 1)/(mod3_lon_2 - mod3_lon_1 + 1)
mod4_var_ts = np.sum(np.sum(mod4_var,axis=1),axis=1)/(mod4_lat_2 - mod4_lat_1 + 1)/(mod4_lon_2 - mod4_lon_1 + 1)

#print(mod2_var_ts)

mod1_var_mask_ts = np.ma.mean(np.ma.mean(mod1_var_mask,axis=1),axis=1)
mod2_var_mask_ts = np.ma.mean(np.ma.mean(mod2_var_mask,axis=1),axis=1)
mod3_var_mask_ts = np.ma.mean(np.ma.mean(mod3_var_mask,axis=1),axis=1)
mod4_var_mask_ts = np.ma.mean(np.ma.mean(mod4_var_mask,axis=1),axis=1)


###########################################################
#calculate statistics
###########################################################

obs1_clim = np.zeros(12)
obs2_clim = np.zeros(12)
obs3_clim = np.zeros(12)
obs4_clim = np.zeros(12)
obs5_clim = np.zeros(12)

obs1_clim_err = np.zeros(12)
obs2_clim_err = np.zeros(12)
obs3_clim_err = np.zeros(12)
obs4_clim_err = np.zeros(12)
obs5_clim_err = np.zeros(12)

obs1_yrts = np.zeros(endyear-iniyear+1)
obs2_yrts = np.zeros(endyear-iniyear+1)
obs3_yrts = np.zeros(endyear-iniyear+1)
obs4_yrts = np.zeros(endyear-iniyear+1)
obs5_yrts = np.zeros(endyear-iniyear+1)

obs1_yrts_err = np.zeros(endyear-iniyear+1)
obs2_yrts_err = np.zeros(endyear-iniyear+1)
obs3_yrts_err = np.zeros(endyear-iniyear+1)
obs4_yrts_err = np.zeros(endyear-iniyear+1)
obs5_yrts_err = np.zeros(endyear-iniyear+1)

mod1_clim = np.zeros(12)
mod2_clim = np.zeros(12)
mod3_clim = np.zeros(12)
mod4_clim = np.zeros(12)

mod1_clim_err = np.zeros(12)
mod2_clim_err = np.zeros(12)
mod3_clim_err = np.zeros(12)
mod4_clim_err = np.zeros(12)

mod1_yrts = np.zeros(endyear-iniyear+1)
mod2_yrts = np.zeros(endyear-iniyear+1)
mod3_yrts = np.zeros(endyear-iniyear+1)
mod4_yrts = np.zeros(endyear-iniyear+1)

mod1_yrts_err = np.zeros(endyear-iniyear+1)
mod2_yrts_err = np.zeros(endyear-iniyear+1)
mod3_yrts_err = np.zeros(endyear-iniyear+1)
mod4_yrts_err = np.zeros(endyear-iniyear+1)

mod1_mask_clim = np.zeros(12)
mod2_mask_clim = np.zeros(12)
mod3_mask_clim = np.zeros(12)
mod4_mask_clim = np.zeros(12)

mod1_mask_clim_err = np.zeros(12)
mod2_mask_clim_err = np.zeros(12)
mod3_mask_clim_err = np.zeros(12)
mod4_mask_clim_err = np.zeros(12)

mod1_mask_yrts = np.zeros(endyear-iniyear+1)
mod2_mask_yrts = np.zeros(endyear-iniyear+1)
mod3_mask_yrts = np.zeros(endyear-iniyear+1)
mod4_mask_yrts = np.zeros(endyear-iniyear+1)

mod1_mask_yrts_err = np.zeros(endyear-iniyear+1)
mod2_mask_yrts_err = np.zeros(endyear-iniyear+1)
mod3_mask_yrts_err = np.zeros(endyear-iniyear+1)
mod4_mask_yrts_err = np.zeros(endyear-iniyear+1)


for idx in range(12):
    obs1_clim[idx] = np.mean(obs1_var_ts[idx::12])
    obs2_clim[idx] = np.mean(obs2_var_ts[idx::12])
    obs3_clim[idx] = np.mean(obs3_var_ts[idx::12])
    obs4_clim[idx] = np.mean(obs4_var_ts[idx::12])
    obs5_clim[idx] = np.mean(obs5_var_ts[idx::12])

    obs1_clim_err[idx] = np.std(obs1_var_ts[idx::12])
    obs2_clim_err[idx] = np.std(obs2_var_ts[idx::12])
    obs3_clim_err[idx] = np.std(obs3_var_ts[idx::12])
    obs4_clim_err[idx] = np.std(obs4_var_ts[idx::12])
    obs5_clim_err[idx] = np.std(obs5_var_ts[idx::12])

    mod1_clim[idx] = np.mean(mod1_var_ts[idx::12])
    mod2_clim[idx] = np.mean(mod2_var_ts[idx::12])
    mod3_clim[idx] = np.mean(mod3_var_ts[idx::12])
    mod4_clim[idx] = np.mean(mod4_var_ts[idx::12])

    mod1_clim_err[idx] = np.std(mod1_var_ts[idx::12])
    mod2_clim_err[idx] = np.std(mod2_var_ts[idx::12])
    mod3_clim_err[idx] = np.std(mod3_var_ts[idx::12])
    mod4_clim_err[idx] = np.std(mod4_var_ts[idx::12])

    mod1_mask_clim[idx] = np.mean(mod1_var_mask_ts[idx::12])
    mod2_mask_clim[idx] = np.mean(mod2_var_mask_ts[idx::12])
    mod3_mask_clim[idx] = np.mean(mod3_var_mask_ts[idx::12])
    mod4_mask_clim[idx] = np.mean(mod4_var_mask_ts[idx::12])

    mod1_mask_clim_err[idx] = np.std(mod1_var_mask_ts[idx::12])
    mod2_mask_clim_err[idx] = np.std(mod2_var_mask_ts[idx::12])
    mod3_mask_clim_err[idx] = np.std(mod3_var_mask_ts[idx::12])
    mod4_mask_clim_err[idx] = np.std(mod4_var_mask_ts[idx::12])

print(obs1_clim)
for idx in range(endyear-iniyear+1):
    obs1_yrts[idx] = np.mean(obs1_var_ts[idx*12:idx*12+12])
    obs2_yrts[idx] = np.mean(obs2_var_ts[idx*12:idx*12+12])
    obs3_yrts[idx] = np.mean(obs3_var_ts[idx*12:idx*12+12])
    obs4_yrts[idx] = np.mean(obs4_var_ts[idx*12:idx*12+12])
    obs5_yrts[idx] = np.mean(obs5_var_ts[idx*12:idx*12+12])

    obs1_yrts_err[idx] = np.std(obs1_var_ts[idx*12:idx*12+12])
    obs2_yrts_err[idx] = np.std(obs2_var_ts[idx*12:idx*12+12])
    obs3_yrts_err[idx] = np.std(obs3_var_ts[idx*12:idx*12+12])
    obs4_yrts_err[idx] = np.std(obs4_var_ts[idx*12:idx*12+12])
    obs5_yrts_err[idx] = np.std(obs5_var_ts[idx*12:idx*12+12])

    mod1_yrts[idx] = np.mean(mod1_var_ts[idx*12:idx*12+12])
    mod2_yrts[idx] = np.mean(mod2_var_ts[idx*12:idx*12+12])
    mod3_yrts[idx] = np.mean(mod3_var_ts[idx*12:idx*12+12])
    mod4_yrts[idx] = np.mean(mod4_var_ts[idx*12:idx*12+12])

    mod1_yrts_err[idx] = np.std(mod1_var_ts[idx*12:idx*12+12])
    mod2_yrts_err[idx] = np.std(mod2_var_ts[idx*12:idx*12+12])
    mod3_yrts_err[idx] = np.std(mod3_var_ts[idx*12:idx*12+12])
    mod4_yrts_err[idx] = np.std(mod4_var_ts[idx*12:idx*12+12])

    mod1_mask_yrts[idx] = np.mean(mod1_var_mask_ts[idx*12:idx*12+12])
    mod2_mask_yrts[idx] = np.mean(mod2_var_mask_ts[idx*12:idx*12+12])
    mod3_mask_yrts[idx] = np.mean(mod3_var_mask_ts[idx*12:idx*12+12])
    mod4_mask_yrts[idx] = np.mean(mod4_var_mask_ts[idx*12:idx*12+12])

    mod1_mask_yrts_err[idx] = np.std(mod1_var_mask_ts[idx*12:idx*12+12])
    mod2_mask_yrts_err[idx] = np.std(mod2_var_mask_ts[idx*12:idx*12+12])
    mod3_mask_yrts_err[idx] = np.std(mod3_var_mask_ts[idx*12:idx*12+12])
    mod4_mask_yrts_err[idx] = np.std(mod4_var_mask_ts[idx*12:idx*12+12])


###########################################################
#draw graphs
###########################################################   
#climatology over continent

plt.figure(1)
#fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
#ax = axs[0,0]

plt.plot(month, mod1_mask_clim, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')
plt.errorbar(month, mod1_mask_clim, yerr = mod1_mask_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.plot(month, mod2_mask_clim, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')
plt.errorbar(month, mod2_mask_clim, yerr = mod2_mask_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'yellow')

plt.plot(month, mod3_mask_clim, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')
plt.errorbar(month, mod3_mask_clim, yerr = mod3_mask_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'green')

plt.plot(month, mod4_mask_clim, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')
plt.errorbar(month, mod4_mask_clim, yerr = mod4_mask_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'blue')

plt.plot(month, obs1_clim, color='black', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'CRU')
plt.errorbar(month, obs1_clim, yerr = obs1_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'black')

plt.plot(month, obs3_clim, color='brown', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'GPCC')
plt.errorbar(month, obs3_clim, yerr = obs3_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'brown')

plt.plot(month, obs5_clim, color='navy', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'APHRODITE')
plt.errorbar(month, obs5_clim, yerr = obs5_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'navy')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' total precip climatology over continent')

plt.savefig(outdir+'vrseasia_prect_mainSEA_clim_ts_overland.pdf')

#clomatology over all grids
plt.figure(2)

plt.plot(month, mod1_clim, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')
plt.errorbar(month, mod1_clim, yerr = mod1_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.plot(month, mod2_clim, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')
plt.errorbar(month, mod2_clim, yerr = mod2_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'yellow')

plt.plot(month, mod3_clim, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')
plt.errorbar(month, mod3_clim, yerr = mod3_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'green')

plt.plot(month, mod4_clim, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')
plt.errorbar(month, mod4_clim, yerr = mod4_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'blue')

plt.plot(month, obs2_clim, color='black', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'GPCP')
plt.errorbar(month, obs2_clim, yerr = obs2_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'black')

plt.plot(month, obs4_clim, color='brown', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'ERA_interim')
plt.errorbar(month, obs4_clim, yerr = obs4_clim_err, fmt='o',markersize=2,elinewidth=1, color = 'brown')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' total precip climatology over all grids')

plt.savefig(outdir+'vrseasia_prect_mainSEA_clim_ts_allgrids.pdf')


#annual ts over continent

plt.figure(3)

plt.plot(yearts, mod1_mask_yrts, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')
plt.errorbar(yearts, mod1_mask_yrts, yerr = mod1_mask_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.plot(yearts, mod2_mask_yrts, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')
plt.errorbar(yearts, mod2_mask_yrts, yerr = mod2_mask_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'yellow')

plt.plot(yearts, mod3_mask_yrts, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')
plt.errorbar(yearts, mod3_mask_yrts, yerr = mod3_mask_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'green')

plt.plot(yearts, mod4_mask_yrts, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')
plt.errorbar(yearts, mod4_mask_yrts, yerr = mod4_mask_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'blue')

plt.plot(yearts, obs1_yrts, color='black', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'CRU')
plt.errorbar(yearts, obs1_yrts, yerr = obs1_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'black')

plt.plot(yearts, obs3_yrts, color='brown', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'GPCC')
plt.errorbar(yearts, obs3_yrts, yerr = obs3_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'brown')

plt.plot(yearts, obs5_yrts, color='navy', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'APHRODITE')
plt.errorbar(yearts, obs5_yrts, yerr = obs5_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'navy')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' annual total precip over continent')

plt.savefig(outdir+'vrseasia_prect_mainSEA_ann_ts_overland.pdf')

#annual ts over all grids

plt.figure(4)

plt.plot(yearts, mod1_yrts, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')
plt.errorbar(yearts, mod1_yrts, yerr = mod1_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'red')

plt.plot(yearts, mod2_yrts, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')
plt.errorbar(yearts, mod2_yrts, yerr = mod2_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'yellow')

plt.plot(yearts, mod3_yrts, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')
plt.errorbar(yearts, mod3_yrts, yerr = mod3_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'green')

plt.plot(yearts, mod4_yrts, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')
plt.errorbar(yearts, mod4_yrts, yerr = mod4_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'blue')

plt.plot(yearts, obs2_yrts, color='black', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'GPCC')
plt.errorbar(yearts, obs2_yrts, yerr = obs2_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'black')

plt.plot(yearts, obs4_yrts, color='brown', marker='o', markersize=2, linestyle='solid',linewidth=2, label = 'ERA_interim')
plt.errorbar(yearts, obs4_yrts, yerr = obs4_yrts_err, fmt='o',markersize=2,elinewidth=1, color = 'brown')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' annual total precip over all grids')

plt.savefig(outdir+'vrseasia_prect_mainSEA_ann_ts_allgrids.pdf')

