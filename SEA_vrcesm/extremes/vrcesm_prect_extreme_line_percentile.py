#this script is used to compare vrcesm against observations
#here extremes is presented
#by Harry Li


#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

#set up cesm data directories and filenames
dir1 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
dir2 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
dir3 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
dir4 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f19_f19_AMIP_1979_to_2005/atm/hist/"
#dir4 = "/scratch/d/dylan/harryli/gpcdata/cesm1.2/gpc_cesm1_2_2/archive/AMIP1979control_2deg/atm/hist/"

fname1 = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
fname2 = "fv09_PREC_ne30_ne30_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
fname3 = "PREC_f09_f09_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
fname4 = "PREC_f19_f19_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
#fname4 = "AMIP1979control_2deg.cam.h1.1979-2005.nc"

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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/extremes/"
outlog = open(outdir+"vrcesm_prect_clim_line_plot_output.log", "w")

#define inital year and end year
iniyear = 1980
endyear = 2005
yearts  = np.arange(iniyear,endyear+1)
#print(yearts)

nyears = (endyear-iniyear+1)
nmonths = nyears*12
ndays = nyears*365
ndaysinyear = 365

#set percentile
pertile = 99

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
#model  
#vrseasia
mod1_lats = mod1_in.variables['lat'][:]
mod1_lons = mod1_in.variables['lon'][:]

mod1_lat_1 = np.argmin( np.abs( mod1_lats - reg_lats[0] ) )
mod1_lat_2 = np.argmin( np.abs( mod1_lats - reg_lats[1] ) )
mod1_lon_1 = np.argmin( np.abs( mod1_lons - reg_lons[0] ) )
mod1_lon_2 = np.argmin( np.abs( mod1_lons - reg_lons[1] ) )

mod1_var = mod1_in.variables['PRECT'][(iniyear-1979)*ndaysinyear : (endyear-1979 + 1) * ndaysinyear, mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]
mod1_var = mod1_var * 86400 *1000
mod1_frc = mod1_lndfrc.variables['LANDFRAC'][mod1_lat_1 : mod1_lat_2 + 1, mod1_lon_1 : mod1_lon_2 + 1]

#ne30
mod2_lats = mod2_in.variables['lat'][:]
mod2_lons = mod2_in.variables['lon'][:]

mod2_lat_1 = np.argmin( np.abs( mod2_lats - reg_lats[0] ) )
mod2_lat_2 = np.argmin( np.abs( mod2_lats - reg_lats[1] ) )
mod2_lon_1 = np.argmin( np.abs( mod2_lons - reg_lons[0] ) )
mod2_lon_2 = np.argmin( np.abs( mod2_lons - reg_lons[1] ) )

mod2_var = mod2_in.variables['PRECT'][(iniyear-1979)*ndaysinyear : (endyear-1979 + 1) * ndaysinyear, mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]
mod2_var = mod2_var * 86400 *1000
mod2_frc = mod2_lndfrc.variables['LANDFRAC'][mod2_lat_1 : mod2_lat_2 + 1, mod2_lon_1 : mod2_lon_2 + 1]

#fv0.9x1.25
mod3_lats = mod3_in.variables['lat'][:]
mod3_lons = mod3_in.variables['lon'][:]

mod3_lat_1 = np.argmin( np.abs( mod3_lats - reg_lats[0] ) )
mod3_lat_2 = np.argmin( np.abs( mod3_lats - reg_lats[1] ) )
mod3_lon_1 = np.argmin( np.abs( mod3_lons - reg_lons[0] ) )
mod3_lon_2 = np.argmin( np.abs( mod3_lons - reg_lons[1] ) )

mod3_var = mod3_in.variables['PRECT'][(iniyear-1979)*ndaysinyear : (endyear-1979 + 1) * ndaysinyear, mod3_lat_1 : mod3_lat_2 + 1, mod3_lon_1 : mod3_lon_2 + 1]
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

#print mod4_lats
#print mod4_lat_1, mod4_lat_2

mod4_var = mod4_in.variables['PRECT'][(iniyear-1979)*ndaysinyear : (endyear-1979 + 1) * ndaysinyear, mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
mod4_var = mod4_var * 86400 *1000
mod4_frc = mod4_lndfrc.variables['LANDFRAC'][mod4_lat_1 : mod4_lat_2 + 1, mod4_lon_1 : mod4_lon_2 + 1]
print(mod4_frc[:,:])
print("-------------------")

#mask the array to fit CRU data
print(mod4_var.shape)
mod1_var_mask = mod1_var.copy()
mod2_var_mask = mod2_var.copy()
mod3_var_mask = mod3_var.copy()
mod4_var_mask = mod4_var.copy()

for idx in range((endyear-iniyear+1)*12):
    if idx==0:
        print(np.ma.masked_where(mod4_frc<0.5, mod4_var[idx,:,:]))
#ax = axs[0,0]
    mod1_var_mask[idx,:,:] = np.ma.masked_where(mod1_frc<0.5, mod1_var_mask[idx,:,:])
    mod2_var_mask[idx,:,:] = np.ma.masked_where(mod2_frc<0.5, mod2_var_mask[idx,:,:])
    mod3_var_mask[idx,:,:] = np.ma.masked_where(mod3_frc<0.5, mod3_var_mask[idx,:,:])
    mod4_var_mask[idx,:,:] = np.ma.masked_where(mod4_frc<0.5, mod4_var_mask[idx,:,:])

print(mod4_var_mask[0,:,:])
print("-------------------")
print(mod4_var[0,:,:])

#calculate time series
mod1_var_ts = np.sum(np.sum(mod1_var,axis=1),axis=1)/(mod1_lat_2 - mod1_lat_1 + 1)/(mod1_lon_2 - mod1_lon_1 + 1)
mod2_var_ts = np.sum(np.sum(mod2_var,axis=1),axis=1)/(mod2_lat_2 - mod2_lat_1 + 1)/(mod2_lon_2 - mod2_lon_1 + 1)
mod3_var_ts = np.sum(np.sum(mod3_var,axis=1),axis=1)/(mod3_lat_2 - mod3_lat_1 + 1)/(mod3_lon_2 - mod3_lon_1 + 1)
mod4_var_ts = np.sum(np.sum(mod4_var,axis=1),axis=1)/(mod4_lat_2 - mod4_lat_1 + 1)/(mod4_lon_2 - mod4_lon_1 + 1)

mod1_var_mask_ts = np.mean(np.mean(mod1_var_mask,axis=1),axis=1)
mod2_var_mask_ts = np.mean(np.mean(mod2_var_mask,axis=1),axis=1)
mod3_var_mask_ts = np.mean(np.mean(mod3_var_mask,axis=1),axis=1)
mod4_var_mask_ts = np.mean(np.mean(mod4_var_mask,axis=1),axis=1)

###########################################################
#calculate extremes
###########################################################

mod1_yrts = np.zeros(endyear-iniyear+1)
mod2_yrts = np.zeros(endyear-iniyear+1)
mod3_yrts = np.zeros(endyear-iniyear+1)
mod4_yrts = np.zeros(endyear-iniyear+1)

mod1_mask_yrts = np.zeros(endyear-iniyear+1)
mod2_mask_yrts = np.zeros(endyear-iniyear+1)
mod3_mask_yrts = np.zeros(endyear-iniyear+1)
mod4_mask_yrts = np.zeros(endyear-iniyear+1)

for idx in range(endyear-iniyear+1):

    mod1_yrts[idx] = np.percentile(mod1_var_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod2_yrts[idx] = np.percentile(mod2_var_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod3_yrts[idx] = np.percentile(mod3_var_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod4_yrts[idx] = np.percentile(mod4_var_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)

    mod1_mask_yrts[idx] = np.percentile(mod1_var_mask_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod2_mask_yrts[idx] = np.percentile(mod2_var_mask_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod3_mask_yrts[idx] = np.percentile(mod3_var_mask_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)
    mod4_mask_yrts[idx] = np.percentile(mod4_var_mask_ts[idx*ndaysinyear:(idx+1)*ndaysinyear],pertile)

###########################################################
#draw graphs
###########################################################
#water extreme threshold ts over continent

plt.figure(1)
#fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
#ax = axs[0,0]

plt.plot(yearts, mod1_mask_yrts, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')

plt.plot(yearts, mod2_mask_yrts, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')

plt.plot(yearts, mod3_mask_yrts, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')

plt.plot(yearts, mod4_mask_yrts, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' '+str(pertile)+' precentile total precip over continent')

plt.savefig(outdir+'vrseasia_prect_mainSEA_extreme_'+str(pertile)+'th_ts_overland.pdf')

#water extreme threshold ts over all grids

plt.figure(4)

plt.plot(yearts, mod1_yrts, color='red', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'vrseasia')

plt.plot(yearts, mod2_yrts, color='yellow', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'ne30')

plt.plot(yearts, mod3_yrts, color='green', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv0.9x1.25')

plt.plot(yearts, mod4_yrts, color='blue', marker='o', markersize=2, linestyle='dashed',linewidth=2, label = 'fv1.9x2.5')

plt.legend()
plt.title(str(iniyear)+'-'+str(endyear)+' '+str(pertile)+' precentile total precip over all grids')

plt.savefig(outdir+'vrseasia_prect_mainSEA_extreme_'+str(pertile)+'th_ts_allgrids.pdf')



