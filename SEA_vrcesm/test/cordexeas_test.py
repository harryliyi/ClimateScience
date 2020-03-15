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
import pandas as pd
import datetime as datetime
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
ignore_years = []
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
yearts = range(iniyear, endyear+1, 1)
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')


print('Reading CORDEX-EAS data...')


dataset = nc.Dataset(
    '/scratch/d/dylan/harryli/obsdataset/CORDEX_EAS/model1/ICHEC-EC-EARTH/HIRHAM5/historical/pr/mon/pr_EAS-44_ICHEC-EC-EARTH_historical_r3i1p1_DMI-HIRHAM5_v1_mon_1951-2005.nc')
time_var = dataset.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    dtime.append(temp)
# dtime  = nc.num2date(time_var[:],time_var.units)
# print(cftime)
# print(dtime)
dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
select_dtime = (dtime >= inidate) & (dtime < enddate) & (
    ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
dtime = dtime[select_dtime]

lats = dataset.variables['lat'][:, 0]
lons = dataset.variables['lon'][0, :]

lat_1 = np.argmin(np.abs(lats - latbounds[0]))
lat_2 = np.argmin(np.abs(lats - latbounds[1]))
lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

var = dataset.variables['pr'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
lats = lats[lat_1: lat_2 + 1]
lons = lons[lon_1: lon_2 + 1]

var = var * 86400 * 1000 / 997

print(lons)
temp = np.mean(np.mean(var, axis=1), axis=1)

monts = np.zeros(12)
for i in range(12):
    monts[i] = np.mean(temp[i::12])
print(temp)
print(monts)
