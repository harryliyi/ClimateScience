'''
# This is a function to read precip from ERA_interim

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime


################################################################################################
# read monthly data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_ERAinterim(varname, iniyear, endyear, varfname, latbounds, lonbounds, oceanmask=0):

    # set up directories dictionary
    dir = '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'

    # set file name
    fname = 'era_interim_'+varfname+'_1979-2016.nc'

    # setup for land fraction
    dir_lndfrc = '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'
    flndfrc = 'era_interim_landsea_mask.nc'
    maskvarname = 'lsm'

    # set up allowed variable names
    varnames = ['skt', 'sp']

    # set up variable name of lat/lon
    latname = 'latitude'
    lonname = 'longitude'

    # set up initial/end date
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # open data file
    dataset = nc.Dataset(dir+fname)

    # read time series
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

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

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

    if varname not in varnames:
        print('Variable does not exit')
        return 0
    else:
        var = dataset.variables[varname][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if oceanmask == 1:
        lsmdata = nc.Dataset(dir_lndfrc+flndfrc)
        lsmask = lsmdata.variables[maskvarname][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask < 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if oceanmask == -1:
        lsmdata = nc.Dataset(dir_lndfrc+flndfrc)
        lsmask = lsmdata.variables[maskvarname][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask > 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    return var, dtime, lats, lons


# read 3D variables at a certain level


def readobs_ERAinterim_wind_3D(varname, iniyear, endyear, varfanme, latbounds, lonbounds, oceanmask=0, **kwargs):

    # set up directories dictionary
    dir = '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'
    # set file name
    fname = 'era_interim_'+varfanme+'_1979-2016.nc'

    # setup for land fraction
    dir_lndfrc = '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'
    flndfrc = 'era_interim_landsea_mask.nc'
    maskvarname = 'lsm'

    # set up variable name of lat/lon
    latname = 'latitude'
    lonname = 'longitude'
    levname = 'level'

    # set up initial/end date
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # open data file
    dataset = nc.Dataset(dir+fname)

    # read time series
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

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]
    levs = dataset.variables[levname][:]

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

    var = dataset.variables[varname][select_dtime, :, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if oceanmask == 1:
        lsmdata = nc.Dataset(dir_lndfrc+flndfrc)
        lsmask = lsmdata.variables[maskvarname][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask < 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if oceanmask == -1:
        lsmdata = nc.Dataset(dir_lndfrc+flndfrc)
        lsmask = lsmdata.variables[maskvarname][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask > 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if lat_swap:
        var = var[:, :, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, :, ::-1]
        lons = lons[::-1]

    return var, dtime, levs, lats, lons

################################################################################################
# test function
################################################################################################


'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'sp'
varfanme = 'ps'
frequency = 'monthly'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_ERAinterim(
    varname, iniyear, endyear, varfanme, frequency, latbounds, lonbounds, oceanmask=0)
var[var.mask] = np.nan
print(lats)
print(lons)
print(time)
print(var.shape)
print(np.mean(np.mean(var, axis=1), axis=1))
'''

'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'z'
varfanme = 'z3'
frequency = 'monthly'
plev = 992

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_ERAinterim_3Dlevel(
    varname, iniyear, endyear, varfanme, frequency, plev, latbounds, lonbounds, oceanmask=0)
var[var.mask] = np.nan
print(lats)
print(lons)
print(time)
print(var.shape)
print(np.mean(np.mean(var, axis=1), axis=1))
'''
