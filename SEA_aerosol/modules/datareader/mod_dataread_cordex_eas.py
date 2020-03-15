'''
# This is a function to read data from CORDEX-SEA data.

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
from mpl_toolkits import basemap
import cf

'''
This is a function used to read data from vrcesm

Description on parameters:
1) varname: the variable name in netCDF

2) iniyear/endyear: the time bounds

3) project: here is SEA-22

4) fequency: monthly, daily or hourly data to read

5) latbounds/lonbounds: the horizontal boundaries of data

6) oceanmask: whether need to mask the ocean, default will not maks the ocean,
    0: do not mask any
    1: maks the ocean
   -1: mask the land

7) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readcordex(varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, **kwargs):

    # read reference grids to interpolate
    refdir = '/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/'
    refname = 'USGS-gtopo30_0.23x0.31_remap_c061107.nc'

    refdata = nc.Dataset(refdir+refname)
    reflats = refdata.variables['lat'][:]
    reflons = refdata.variables['lon'][:]
    lat_1 = np.argmin(np.abs(reflats - latbounds[0]))
    lat_2 = np.argmin(np.abs(reflats - latbounds[1]))
    lon_1 = np.argmin(np.abs(reflons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(reflons - lonbounds[1]))
    reflats = reflats[lat_1: lat_2 + 1]
    reflons = reflons[lon_1: lon_2 + 1]

    models = {
        'model1': 'ICHEC-EC-EARTH_historical_r3i1p1_DMI-HIRHAM5_v1',
        'model2': 'MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-CCLM5-0-2_v1',
        'model3': 'ICHEC-EC-EARTH_historical_r12i1p1_CLMcom-CCLM5-0-2_v1',
        'model4': 'CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM5-0-2_v1',
        'model5': 'MOHC-HadGEM2-ES_historical_r1i1p1_CLMcom-CCLM5-0-2_v1'
    }

    gcms = {
        'model1': 'ICHEC-EC-EARTH',
        'model2': 'MPI-M-MPI-ESM-LR',
        'model3': 'ICHEC-EC-EARTH',
        'model4': 'CNRM-CERFACS-CNRM-CM5',
        'model5': 'MOHC-HadGEM2-ES'
    }

    rcms = {
        'model1': 'HIRHAM5',
        'model2': 'CCLM5-0-2',
        'model3': 'CCLM5-0-2',
        'model4': 'CCLM5-0-2',
        'model5': 'CCLM5-0-2'
    }

    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    fdir = '/scratch/d/dylan/harryli/obsdataset/CORDEX_EAS/' + modelname+'/' + \
        gcms[modelname]+'/'+rcms[modelname]+'/historical/'+varname+'/'+frequency+'/'

    fname = varname+'_'+project+'_'+models[modelname]+'_'+frequency+'_1951-2005.nc'
    dataset = nc.Dataset(fdir+fname)
    f = cf.read(fdir+fname)[0]
    print(f)
    cflats = cf.DimensionCoordinate(data=cf.Data(reflats, 'degrees_north'))
    cflons = cf.DimensionCoordinate(data=cf.Data(reflons, 'degrees_east'))
    if modelname == 'model1':
        h = f.regrids({'latitude': cflats, 'longitude': cflons}, 'bilinear', src_axes={'X': 'projection_x_coordinate', 'Y': 'projection_y_coordinate'}, src_cyclic=False)
    else:
        h = f.regrids({'latitude': cflats, 'longitude': cflons}, 'bilinear', src_axes={'X': 'grid_longitude', 'Y': 'grid_latitude'}, src_cyclic=False)
    print(h)

    time_var = dataset.variables['time']
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)

    dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(select_dtime)
    # print(dtime[0:365])

    var = h.data.array
    print(h.data)
    print(var.shape)
    var = np.ma.array(var[select_dtime, :, :])

    lats = dataset.variables['lat'][:, :]
    lons = dataset.variables['lon'][:, :]

    return var, dtime, lats, lons, reflats, reflons


# test on the function
varname = 'pr'
iniyear = 2000
endyear = 2005
project = 'EAS-44'
modelname = 'model1'
frequency = 'mon'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons, reflats, reflons = readcordex(varname, iniyear, endyear, project, modelname,
                                                     frequency, latbounds, lonbounds, oceanmask=0)
var[var.mask] = np.nan
var = var * 86400 * 1000 / 997
print(lats)
print(lons)
print(time)
print(var.shape)
# print(var[0, :, :])
temp = np.mean(np.mean(var, axis=1), axis=1)
monts = np.zeros(12)
mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for i in range(12):
    if modelname == 'model1':
        monts[i] = np.mean(temp[i::12])
    else:
        monts[i] = np.mean(temp[i::12])/mdays[i]
print(monts)
