# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-plot basic analysis
# S3-calculate and plot extreme
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cordex_sea import readcordex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
import pickle
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/SA-OBS/clustering/'
kmeans_resdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/SA-OBS/clustering/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
# print(yearts)

# select number of clusters
ncluster = 3

# define regions
# latbounds = [-15, 25]
# lonbounds = [90, 145]
latbounds = [10, 25]
lonbounds = [100, 110]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'mon'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

############################################################################
# read data
############################################################################

print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

print(cordex_time1)
print(cordex_var1)

var = np.mean(np.mean(cordex_var1, axis=1), axis=1)
print(cordex_lats1)
print(cordex_lats2)
print(cordex_lats3)
print(cordex_lats4)
# print(var)

dataset = nc.Dataset(
    '/scratch/d/dylan/harryli/obsdataset/CORDEX_SEA/IPSL-IPSL-CM5A-LR/historical/pr/mon/pr_SEA-22_IPSL-IPSL-CM5A-LR_historical_r1i1p1_ICTP-RegCM4-3_v4_mon_2005.nc')
dataset = nc.Dataset(
    '/scratch/d/dylan/harryli/obsdataset/CORDEX_SEA/IPSL-IPSL-CM5A-LR/historical/pr/mon/pr_SEA-22_IPSL-IPSL-CM5A-LR_historical_r1i1p1_ICTP-RegCM4-3_v4_mon_2005.nc')

lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]
print(lons)
