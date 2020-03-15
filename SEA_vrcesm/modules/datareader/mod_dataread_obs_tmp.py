'''
# This is a function to read precip from CRU, GPCP, GPCC, ERA-interim, APHRODITE

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
from calendar import monthrange


################################################################################################
# read monthly data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) project: the name of dataset

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_tmp_mon(project, iniyear, endyear, latbounds, lonbounds):

    # set up directories dictionary
    dirs = {'CRU': '/scratch/d/dylan/harryli/obsdataset/CRU/tmp/',
            'ERA-interim': '/scratch/d/dylan/harryli/obsdataset/ERA_interim/tmp/surface/monthly/',
            'GHCN-CAMS': '/scratch/d/dylan/harryli/obsdataset/GHCN/tmp/GHCNCAMS/',
            'University of Delaware': '/scratch/d/dylan/harryli/obsdataset/UDel/'}

    maskdirs = {'ERA-interim': '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'}

    maskfnames = {'ERA-interim': 'era_interim_landsea_mask.nc'}

    maskvarnames = {'ERA-interim': 'lsm'}

    fnames = {'CRU': 'cru_ts3.24.01.1901.2015.tmp.dat.nc',
              'ERA-interim': 'era_interim_t2m_monthly_197901-200512.nc',
              'GHCN-CAMS': 'GHCNCAMS.air.mon.mean.nc',
              'University of Delaware': 'UDel.air.mon.mean.v501.nc'}

    varnames = {'CRU': 'tmp',
                'ERA-interim': 't2m',
                'GHCN-CAMS': 'air',
                'University of Delaware': 'air'}

    latnames = {'CRU': 'lat',
                'ERA-interim': 'latitude',
                'GHCN-CAMS': 'lat',
                'University of Delaware': 'lat'}

    lonnames = {'CRU': 'lon',
                'ERA-interim': 'longitude',
                'GHCN-CAMS': 'lon',
                'University of Delaware': 'lon'}

    factors = {'CRU': 0.0,
               'ERA-interim': 273.15,
               'GHCN-CAMS': 273.15,
               'University of Delaware': 0.0}

    mdays = []

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    dataset = nc.Dataset(dirs[project]+fnames[project])

    time_var = dataset.variables['time']
    if hasattr(time_var, 'calendar'):
        cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    else:
        cftime = nc.num2date(time_var[:], units=time_var.units)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    select_dtime = (dtime >= inidate) & (dtime < enddate)
    dtime = dtime[select_dtime]
    # print(dtime)
    # print(select_dtime)

    lats = dataset.variables[latnames[project]][:]
    lons = dataset.variables[lonnames[project]][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    lat_swap = False
    lon_swap = False

    if lat_1 > lat_2:
        lat_swap = True
        temp = lat_1
        lat_1 = lat_2
        lat_2 = temp
    if lon_1 > lon_2:
        lon_swap = True
        temp = lon_1
        lon_1 = lon_2
        lon_2 = temp

    var = dataset.variables[varnames[project]][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    var = var - factors[project]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if project == 'ERA-interim':
        lsmdata = nc.Dataset(maskdirs[project]+maskfnames[project])
        lsmask = lsmdata.variables[maskvarnames[project]][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask < 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    return var, dtime, lats, lons

################################################################################################
# test on the function
################################################################################################


'''
iniyear = 2000
endyear = 2005
project = 'ERA-interim'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_tmp_mon(project, iniyear, endyear, latbounds, lonbounds)
var[var.mask] = np.nan
print(lats)
print(lons)
print(time)
print(var.shape)
# print(var[0, :, :])
print(np.mean(np.mean(var, axis=1), axis=1))
'''
