#this script is used to read data from vrcesm output that is in netcdf format
#by Harry Li

#import libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#define initial year and end year
iniyear = 1995
endyear = 2004

#define vrcesm directory 
cesmdir = "/scratch/d/djones/harryli/cesm1.2/VRcesm/cesm1_2_2/archive/AMIP_VRcesm_diag_1992_2005/atm/hist/"
cesmfile = "fv02_PRECT_AMIP_VRcesm_diag_1992_2005.cam.h0.1992-2004.nc"

#read the data
dataset = Dataset(cesmdir+cesmfile)

print (dataset.dimensions.keys())
print (dataset.dimensions['time'])
print (dataset.variables.keys())

lats = dataset.variables['lat'][:] 
lons = dataset.variables['lon'][:]

darwin = {'name': 'Darwin, Australia', 'lat': -12.45, 'lon': 130.83}

lat_idx = np.abs(lats - darwin['lat']).argmin()
lon_idx = np.abs(lons - darwin['lon']).argmin()

pre = dataset.variables['PRECT'][(iniyear-1992)*12:(endyear-1992+1)*12,lat_idx, lon_idx]

pre = pre *86400*1000
print(pre)
print(len(pre))

x=np.arange(120)
print(x)
print(x[0::12])

date = np.array('1995-01', dtype=np.datetime64)
date = date + np.arange(120)

print(date)

plt.plot(date,pre)
plt.savefig('clean data.pdf')
